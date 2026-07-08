# arXiv:2312.12436v2[cs.CV]20Dec2023

## A Challenger to GPT-4V? Early Explorations of Gemini in Visual Expertise

Chaoyou Fu1∗♠, Renrui Zhang2,3∗, Zihan Wang6∗, Yubo Huang4, Zhengye Zhang4 Longtian Qiu2, Gaoxiang Ye5, Yunhang Shen1, Mengdan Zhang1 Peixian Chen1, Sirui Zhao4, Shaohui Lin6, Deqiang Jiang1 Di Yin1, Peng Gao2, Ke Li1, Hongsheng Li3, Xing Sun1†

1Tencent Youtu Lab, 2Shanghai AI Laboratory 3CUHK MMLab, 4USTC, 5Peking University, 6ECNU

∗ Equal Contribution ♠ Project Leader † Corresponding Author

###### Abstract

The surge of interest towards Multi-modal Large Language Models (MLLMs), e.g., GPT-4V(ision) from OpenAI, has marked a significant trend in both academia and industry. They endow Large Language Models (LLMs) with powerful capabilities in visual understanding, enabling them to tackle diverse multi-modal tasks. Very recently, Google released Gemini, its newest and most capable MLLM built from the ground up for multi-modality. In light of the superior reasoning capabilities, can Gemini challenge GPT-4V’s leading position in multi-modal learning? In this paper, we present a preliminary exploration of Gemini Pro’s visual understanding proficiency, which comprehensively covers four domains: fundamental perception, advanced cognition, challenging vision tasks, and various expert capacities. We compare Gemini Pro with the state-of-the-art GPT-4V to evaluate its upper limits, along with the latest open-sourced MLLM, Sphinx, which reveals the gap between manual efforts and black-box systems. The qualitative samples indicate that, while GPT-4V and Gemini showcase different answering styles and preferences, they can exhibit comparable visual reasoning capabilities, and Sphinx still trails behind them concerning domain generalizability. Specifically, GPT-4V tends to elaborate detailed explanations and intermediate steps, and Gemini prefers to output a direct and concise answer. The quantitative evaluation on the popular MME benchmark, which is specifically designed for MLLM, also demonstrates the impressive multi-modal understanding performance of Gemini, and its potential to be a strong challenger to GPT-4V. Our early investigation of Gemini also observes some common issues of MLLMs concerning visual understanding, logical reasoning, and prompting robustness, indicating that there still remains a considerable distance towards artificial general intelligence. We hope this report may cast a new light on future MLLM research and application scenarios. Our project for tracking the progress of MLLM is released at https://github.

com/BradyFU/Awesome-Multimodal-Large-Language-Models.

###### Contents

List of Figures 3

- 1 Introduction 6

- 1.1 Motivation and Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 1.2 Evaluation Suite . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 1.2.1 Prompt Technique . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 1.2.2 Sample Collection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

###### 2 Fundamental Perception 8

- 2.1 Object-Centric Perception . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 2.2 Scene-Level Perception . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 2.3 Knowledge-based Perception . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

###### 3 Advanced Cognition 48

- 3.1 Text-Rich Visual Reasoning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
- 3.2 Abstract Visual Reasoning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
- 3.3 Science Problem-Solving . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49
- 3.4 Emotion Understanding . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49
- 3.5 Game Playing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49

###### 4 Vision Task 89

- 4.1 Image-Level Vision Task . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 89
- 4.2 Temporal-Level Vision Task . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 89

###### 5 Expert Capacity 100

- 5.1 Autonomous Driving . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 100
- 5.2 Defect Detection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 100
- 5.3 Medical Diagnosis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 100
- 5.4 Economic Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 100
- 5.5 Surveillance and Security . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 101
- 5.6 Remote Sensing Image Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . 101
- 5.7 Robot Motion Planning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 101

###### 6 Quantitative Experiments 121

- 6.1 MME Benchmark . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 121
- 6.2 Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 121

###### 7 Conclusion 123

- 7.1 Summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 123
- 7.2 Gemini vs GPT-4V . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 123
- 7.3 Gemini vs Sphinx . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 123
- 7.4 Future Directions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 124

###### List of Figures

- 1 Section 2.1: spatial relation recognition. . . . . . . . . . . . . . . . . . . . . . . . 10
- 2 Section 2.1: spatial relation recognition. . . . . . . . . . . . . . . . . . . . . . . . 11
- 3 Section 2.1: object counting. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- 4 Section 2.1: difference spotting. . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- 5 Section 2.1: difference spotting. . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 6 Section 2.1: optical illusion recognition. . . . . . . . . . . . . . . . . . . . . . . . 15
- 7 Section 2.1: optical illusion recognition. . . . . . . . . . . . . . . . . . . . . . . . 16
- 8 Section 2.2: scene understanding from image. . . . . . . . . . . . . . . . . . . . . 17
- 9 Section 2.2: scene understanding from image. . . . . . . . . . . . . . . . . . . . . 18
- 10 Section 2.2: scene understanding from image. . . . . . . . . . . . . . . . . . . . . 19
- 11 Section 2.2: scene understanding from video. . . . . . . . . . . . . . . . . . . . . 20
- 12 Section 2.3: commonsense. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- 13 Section 2.3: commonsense. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
- 14 Section 2.3: commonsense. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
- 15 Section 2.3: commonsense. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- 16 Section 2.3: scientific knowledge. . . . . . . . . . . . . . . . . . . . . . . . . . . 25
- 17 Section 2.3: scientific knowledge. . . . . . . . . . . . . . . . . . . . . . . . . . . 26
- 18 Section 2.3: scientific knowledge. . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- 19 Section 2.3: historical knowledge. . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- 20 Section 2.3: multicultural customs. . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- 21 Section 2.3: multicultural customs. . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- 22 Section 2.3: multicultural customs. . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- 23 Section 2.3: celebrity recognition and description. . . . . . . . . . . . . . . . . . . 32
- 24 Section 2.3: celebrity recognition and description. . . . . . . . . . . . . . . . . . . 33
- 25 Section 2.3: landmark recognition and description. . . . . . . . . . . . . . . . . . 34
- 26 Section 2.3: landmark recognition and description. . . . . . . . . . . . . . . . . . 35
- 27 Section 2.3: logo recognition and description. . . . . . . . . . . . . . . . . . . . . 36
- 28 Section 2.3: logo recognition and description. . . . . . . . . . . . . . . . . . . . . 37
- 29 Section 2.3: movie recognition and description. . . . . . . . . . . . . . . . . . . . 38
- 30 Section 2.3: movie recognition and description. . . . . . . . . . . . . . . . . . . . 39
- 31 Section 2.3: movie recognition and description. . . . . . . . . . . . . . . . . . . . 40
- 32 Section 2.3: food recognition and description. . . . . . . . . . . . . . . . . . . . . 41
- 33 Section 2.3: food recognition and description. . . . . . . . . . . . . . . . . . . . . 42
- 34 Section 2.3: plant recognition. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- 35 Section 2.3: animal and plant recognition. . . . . . . . . . . . . . . . . . . . . . . 44
- 36 Section 2.3: artwork recognition and description. . . . . . . . . . . . . . . . . . . 45
- 37 Section 2.3: artwork recognition and description. . . . . . . . . . . . . . . . . . . 46
- 38 Section 2.3: artwork recognition and description. . . . . . . . . . . . . . . . . . . 47

- 39 Section 3.1: table and chart reasoning. . . . . . . . . . . . . . . . . . . . . . . . . 51
- 40 Section 3.1: table and chart reasoning. . . . . . . . . . . . . . . . . . . . . . . . . 52
- 41 Section 3.1: table and chart reasoning. . . . . . . . . . . . . . . . . . . . . . . . . 53
- 42 Section 3.1: table and chart reasoning. . . . . . . . . . . . . . . . . . . . . . . . . 54
- 43 Section 3.1: table and chart reasoning. . . . . . . . . . . . . . . . . . . . . . . . . 55
- 44 Section 3.1: visual code generation. . . . . . . . . . . . . . . . . . . . . . . . . . 56
- 45 Section 3.1: visual code generation. . . . . . . . . . . . . . . . . . . . . . . . . . 57
- 46 Section 3.1: visual code generation. . . . . . . . . . . . . . . . . . . . . . . . . . 58
- 47 Section 3.1: visual code generation. . . . . . . . . . . . . . . . . . . . . . . . . . 59
- 48 Section 3.1: abstract visual stimuli. . . . . . . . . . . . . . . . . . . . . . . . . . . 60
- 49 Section 3.2: abstract visual stimuli. . . . . . . . . . . . . . . . . . . . . . . . . . . 61
- 50 Section 3.2: Wechsler Adult Intelligence Scale. . . . . . . . . . . . . . . . . . . . 62
- 51 Section 3.2: Wechsler Adult Intelligence Scale. . . . . . . . . . . . . . . . . . . . 63
- 52 Section 3.2: Raven’s Progressive Matrices. . . . . . . . . . . . . . . . . . . . . . . 64
- 53 Section 3.2: Raven’s Progressive Matrices. . . . . . . . . . . . . . . . . . . . . . . 65
- 54 Section 3.3: mathematical problem. . . . . . . . . . . . . . . . . . . . . . . . . . 66
- 55 Section 3.3: mathematical problem. . . . . . . . . . . . . . . . . . . . . . . . . . 67
- 56 Section 3.3: mathematical problem. . . . . . . . . . . . . . . . . . . . . . . . . . 68
- 57 Section 3.3: mathematical problem. . . . . . . . . . . . . . . . . . . . . . . . . . 69
- 58 Section 3.3: mathematical problem. . . . . . . . . . . . . . . . . . . . . . . . . . 70
- 59 Section 3.3: mathematical problem. . . . . . . . . . . . . . . . . . . . . . . . . . 71
- 60 Section 3.3: physics problem. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 72
- 61 Section 3.3: physics problem. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 73
- 62 Section 3.3: physics problem. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
- 63 Section 3.4: facial expression analysis. . . . . . . . . . . . . . . . . . . . . . . . . 75
- 64 Section 3.4: facial expression analysis. . . . . . . . . . . . . . . . . . . . . . . . . 76
- 65 Section 3.4: image emotion analysis. . . . . . . . . . . . . . . . . . . . . . . . . . 77
- 66 Section 3.4: image emotion analysis. . . . . . . . . . . . . . . . . . . . . . . . . . 78
- 67 Section 3.4: image emotion analysis. . . . . . . . . . . . . . . . . . . . . . . . . . 79
- 68 Section 3.4: image emotion analysis. . . . . . . . . . . . . . . . . . . . . . . . . . 80
- 69 Section 3.4: image emotion analysis. . . . . . . . . . . . . . . . . . . . . . . . . . 81
- 70 Section 3.4: emotion-conditioned output. . . . . . . . . . . . . . . . . . . . . . . 82
- 71 Section 3.4: emotion-conditioned output. . . . . . . . . . . . . . . . . . . . . . . 83
- 72 Section 3.5: Sudoku. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 84
- 73 Section 3.5: Crossword. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 85
- 74 Section 3.5: Crossword. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 86
- 75 Section 3.5: Go playing. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 87
- 76 Section 3.5: Go playing. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 88
- 77 Section 4.1: object detection. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 90

- 78 Section 4.1: referring expression comprehension. . . . . . . . . . . . . . . . . . . 91
- 79 Section 4.1: referring expression comprehension. . . . . . . . . . . . . . . . . . . 92
- 80 Section 4.1: phrase localization. . . . . . . . . . . . . . . . . . . . . . . . . . . . 93
- 81 Section 4.1: face detection and recognition. . . . . . . . . . . . . . . . . . . . . . 94
- 82 Section 4.2: object tracking. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 95
- 83 Section 4.2: video action recognition. . . . . . . . . . . . . . . . . . . . . . . . . 96
- 84 Section 4.2: video action recognition. . . . . . . . . . . . . . . . . . . . . . . . . 97
- 85 Section 4.2: video action recognition. . . . . . . . . . . . . . . . . . . . . . . . . 98
- 86 Section 4.2: visual story generation. . . . . . . . . . . . . . . . . . . . . . . . . . 99
- 87 Section 5.1: traffic signs understanding. . . . . . . . . . . . . . . . . . . . . . . . 102
- 88 Section 5.1: traffic signs understanding. . . . . . . . . . . . . . . . . . . . . . . . 103
- 89 Section 5.1: driving intentions. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
- 90 Section 5.1: driving intentions. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 105
- 91 Section 5.2: defect detection. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 106
- 92 Section 5.2: defect detection. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 107
- 93 Section 5.2: defect detection. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
- 94 Section 5.3: medical image understanding. . . . . . . . . . . . . . . . . . . . . . . 109
- 95 Section 5.3: medical image understanding. . . . . . . . . . . . . . . . . . . . . . . 110
- 96 Section 5.3: medical image understanding. . . . . . . . . . . . . . . . . . . . . . . 111
- 97 Section 5.3: medical image understanding. . . . . . . . . . . . . . . . . . . . . . . 112
- 98 Section 5.4: economic analysis. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 113
- 99 Section 5.4: economic analysis. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 114
- 100 Section 5.5: surveillance and security. . . . . . . . . . . . . . . . . . . . . . . . . 115
- 101 Section 5.5: surveillance and security. . . . . . . . . . . . . . . . . . . . . . . . . 116
- 102 Section 5.6: remote sensing image analysis. . . . . . . . . . . . . . . . . . . . . . 117
- 103 Section 5.6: remote sensing image analysis. . . . . . . . . . . . . . . . . . . . . . 118
- 104 Section 5.7: robot motion planning. . . . . . . . . . . . . . . . . . . . . . . . . . 119
- 105 Section 5.7: robot motion planning. . . . . . . . . . . . . . . . . . . . . . . . . . 120
- 106 Section 6.2: evaluation on MME benchmark. . . . . . . . . . . . . . . . . . . . . 122

###### 1 Introduction

###### 1.1 Motivation and Overview

Driven by big data and substantial computational power, the realm of large language models (LLMs) [8, 42, 13, 3, 50, 27] has garnered unprecedented enthusiasm and advancement, showcasing generalizability in a wide range of fields. Building upon this achievement, Multi-modal Large Language Models (MLLMs) [61, 30, 16, 5, 23, 65, 36, 14, 60, 25, 20, 6] are emerging as a focal point of research in the new generation. They target on incorporating LLMs with additional sensory input, e.g., image [36, 63], audio [56], 3D [24], etc. Conditioned on data of new modalities, MLLMs take a significant step forward on the path towards general artificial intelligence. Notably, GPT4V(ision) [43, 1, 59] from OpenAI is recognized as the most powerful MLLMs to date, surpassing a host of LLaMA-based [50] models, e.g., LLaMA-Adapter [63], LLaVA [36], and MiniGPT-4 [65]. However, very recently released by Google, Gemini [21] has emerged as a formidable challenger to GPT-4V, which exhibits significant multi-modal capabilities over different benchmarks [19, 38, 26]. Given that the full potential of Gemini has not yet been fully tapped, in this paper, we conduct an early exploration by comparing Gemini with existing best-performing MLLM, i.e., GPT-4V, to reveal its multi-modal capabilities.

For a comprehensive evaluation, we carefully collect a bunch of qualitative samples covering different domains in multi-modal understanding. Two existing representative MLLMs are selected as baselines. The first is GPT-4V, representing the current highest standard in the field, which assesses the upper limits of Gemini. The second is Sphinx [35], a state-of-the-art LLaMA-based MLLM, exploring how much the performance gap is between open-sourced models and closed-sourced systems. Specifically, the qualitative samples can be categorized into four visual domains as follows:

- 1. Fundamental Perception. (Section 2) This dimension focuses on the basic ability of MLLMs to perceive and understand visual concepts, without the need for complex reasoning. It can be subdivided into three key aspects: object-centric, scene-level, and knowledge-based perception. Therein, object-centric perception assesses the model’s capacity to recognize and interpret the characteristics of individual objects within a visual context, exemplified by tasks such as spatial relation recognition, object counting, difference spotting, etc. In contrast, scene-level perception evaluates the understanding of entire scenes from a global perspective, demonstrating the model’s proficiency in image and video captioning. Finally, knowledge-based perception reveals the model’s accumulation and application of specific knowledge across various domains. It encompasses commonsense knowledge, scientific knowledge, cultural customs, and world memory, which respectively cover the content of everyday scenarios, academic disciplines, cultural practices, and global entities.
- 2. Advanced Cognition. (Section 3) The samples in advanced cognition require MLLMs to process more complicated visual information and conduct multi-modal reasoning for problem-solving. The related tasks include text-rich and abstract visual reasoning, science problem solving, emotion understanding, and game playing. Text-rich tasks evaluate the OCR performance of textual content for table and chart reasoning, and the code generation capability conditioned on different visual inputs. Abstract visual reasoning refers to the non-verbal test assessing general intelligence and abstract reasoning, such as the Wechsler Adult Intelligence Scale and Raven’s Progressive Matrices. Science problem-solving, e.g., mathematics and physics, has become a vital metric for measuring MLLMs’ comprehension of quantitative and logical knowledge, involving complex multi-step and chain-of-thought (CoT) reasoning. Moreover, emotion understanding focuses on the detection of underlying emotional information within visual contexts, and game playing evaluates strategic thinking and rule-following abilities in games like Sudoku.
- 3. Challenging Vision Tasks. (Section 4) In this part, we aim to evaluate how MLLMs perform in some challenging vision tasks beyond general visual question-answering, such as object detection, referring expression comprehension, phrase localization, video temporal reasoning, and so on. These tasks require the in-depth visual perception and understanding capabilities of MLLMs. The performance of MLLMs can indicate their potential to serve as multi-purpose vision generalists.
- 4. Expert Capacity. (Section 5) The final dimension evaluates the model’s proficiency in several specialized fields. The scenarios include medical imaging, defect detection, stock

prediction, autonomous driving, and surveillance security. Each of these areas tests the model’s application of its learned knowledge and cognitive skills in a professional context, such as diagnosing diseases from medical images or predicting market trends in stock trading. This demonstrates the generalization capacity of MLLMs from more diverse perspectives.

Besides qualitative samples, we report quantitative results of Gemini on the popular MME benchmark [19] in Section 6, which comprehensively evaluates MLLMs in 14 subtasks from both perception and cognition perspectives.

- 1.2 Evaluation Suite

- 1.2.1 Prompt Technique

GPT-4V has been demonstrated to support a diverse range of prompt techniques [59], from simple instruction following [44, 41, 53, 48] to in-context few-shot learning [8, 51, 2]. This inspires us to design the following prompt techniques. Simple instruction following. A simple instruction directly expresses the user’s intention, such as “Describe this image” or “Who is this person in the poster?”. Existing MLLMs [23, 65, 36, 14, 60] are generally capable of following instructions, enabling us to utilize the simple instruction to accomplish most tasks effectively. We adopt simple instructions to prompt models on most of the tasks. Figures 1 and 3 are typical examples, respectively.

Visual referring prompt. In many cases, a simple visual marker can more effectively convey the user’s interest in a specific spatial region to MLLMs than detailed and lengthy text, as shown in Figure 82. In addition to the visual markers used as visual prompts in [12, 59], we also experiment with physical objects to guide the model’s understanding of the referring items, such as a finger or a pen, as illustrated in the bottom part of Figure 35. Compared to prompting the model with visual markers, using real objects as prompts is more practical in real-time interaction scenarios.

Chain-of-Thought (CoT) prompt. For problems involving complex logical reasoning, we use CoT techniques [54, 32] to guide the model to provide a final response through a series of more logical thought processes, which is shown in Figure 42.

In-context few-shot learning. In certain scenarios where simple text instructions fail to completely demonstrate the task, we employ in-context few-shot learning [8, 51, 2] for better prompting. By providing a few in-context examples at inference time, the model can infer intentions from these examples, thus facilitating the generation of the desired outputs, which is shown in Figure 2.

- 1.2.2 Sample Collection

Avoiding sample leakage. We endeavor to ensure that the collected qualitative images and text are unseen by the models to prevent responses that merely reflect memories of the training data. All the texts in the query are constructed from scratch. The image sources include manually created drawings, offline photographs, Internet images, and some existing datasets [10, 7, 15, 49]. For the Internet images, we strive to collect those with timestamps postdating November 2023.

Diverse difficulty. For each task, we collect samples of varying difficulty levels, e.g., from fundamental perception and cognition to the more challenging vision and expert tasks. In this way, we can not only demonstrate the potential of MLLMs to complete the tasks, but also touch their ability boundaries through some obvious mistake patterns.

###### 2 Fundamental Perception

Fundamental perception, in the context of multi-modal large models, refers to the model’s ability to process and interpret sensory data, primarily visual, to create a coherent understanding of the environment it perceives. The proficiency in perception directly influences a model’s capability in higher-order tasks, as it determines how accurately and effectively the model can acquire and process raw visual input.

In Section 2.1, we will explore the object-centric perception task, such as spatial relationship recognition, object counting, and difference spotting. In Section 2.2, we will examine the models’ capacity for interpreting the entire scenes on diverse domains. In Section 2.3, we will investigate the models’ ability to comprehend visual information via the application of knowledge, which encompasses commonsense, subject knowledge, multicultural customs, and world memory.

###### 2.1 Object-Centric Perception

Spatial relationship recognition. We investigate the models’ capability to comprehend spatial relationships. We find that it seems difficult for the models to identify left and right. For instance, in Figure 1, the individual on the left-hand side of Jordan is James. However, the responses from Gemini and GPT4-V are both Kobe, while Sphinx’s response is Jordan. In our endeavor, we employ in-context few-shot learning techniques to aid the model in comprehending the concept of ‘left-hand’. As depicted in Figure 2, we provide two image examples to instruct the model on what constitutes the ‘left-hand’. However, only GPT-4V successfully learns the concept, while Gemini and Sphinx still can not distinguish between left and right.

Object counting. Figure 3 shows the models’ ability to count objects. It is observed that for simple samples, the performance of the open-source model Sphinx closely aligns with that of the two closedsource models, which is shown in the first three cases. However, as shown in the fourth example, when the images contain an excess of visual elements, all three models tend to make mistakes.

Difference spotting. In Figures 4-5, we present the model’s capacity to spot differences in cartoon images, sketches, and actual photographs. We observe that all models possess the potential to perceive the fine-grained differences between images, although their performance is not consistently stable. In addition, we observe that both Gemini and GPT-4V are easily misled by the intentionally erroneous prompts we provide. As shown in Figure 5, there are actually only three differences. However, when we request the models to identify five differences, both Gemini and GPT-4V fabricate five distinct points and respond incorrectly.

Optical illusion recognition. As shown in Figures 6-7, we investigate whether these models exhibit a visual understanding of optical illusions similar to that of humans. For instance, in the left part of Figure 6, the two pears actually possess identical brightness. However, the interspersed black and white stripes create an illusion, making the pear on the right appear brighter. Gemini recognizes that the two have the same brightness, whereas GPT-4V and Sphinx, like many humans, are deceived by the optical illusion, perceiving the right pear to be brighter. In the right section of Figure 6, GPT-4V identifies a similarity in the angles of the tree trunks and branches to those of human bodies and arms, once again demonstrating a human-like visual understanding of optical illusions.

###### 2.2 Scene-Level Perception

Scene understanding from image. We prompt the models to identify all visual elements in the image as detailed as possible via the text query “Describe this image in detail.” Figures 8-10 illustrate that all three models are capable of depicting the key visual elements within the scene. However, in comparison, GPT-4V shows superior performance, particularly in highly cluttered environments. This is evident in Figure 8, where GPT-4V’s descriptions are notably more detailed and exhibit fewer instances of hallucination.

Scene understanding from video. Here we examine the potential of the models to understand scenes from video. As shown in Figure 11, we extract three temporally distinct frames from a video and input them into the model along with the text query, “Please describe this scene according to these temporal images.” Our observations indicate that Gemini is capable of integrating the information from the different frames into a cohesive scene description. Especially, the first frame displays two round

tables and one potted plant, while the second frame shows one round table and three potted plants. Remarkably, Gemini successfully merges the information from both frames to accurately describe the scene as containing two round tables and three potted plants. GPT-4V describes the contents of images frame by frame. In contrast, Sphinx’s descriptions do not demonstrate a comprehensive understanding of the sequence of images.

###### 2.3 Knowledge-based Perception

Commonsense. Figures 12-15 illustrate the capability of these three models to apply common sense in understanding visual information within images. It is observed that the open-source model Sphinx performs comparably to Gemini and GPT-4V in applying social norms as shown in Figures 12-13. However, it exhibits a slightly inferior performance in the application of physical laws. For instance, as shown in Figure 14, Gemini and GPT-4V can accurately select heavy clothing for cold weather protection in Antarctica. Interestingly, while Sphinx recognizes the need for thick clothing in cold weather, it erroneously identifies the image of a T-shirt as suitable for cold protection.

Subject knowledge. In Figures 16-19, we examine the model’s ability to apply knowledge in the fields of physics, chemistry, and history. We observe that both Gemini and GPT-4V possess the relevant subject knowledge associated with these cases. The performance of Sphinx is slightly inferior, yet it is capable of providing correct answers in certain instances.

Multicultural customs. We examine the models’ comprehension of multicultural elements. We present the model with images featuring distinct local ethnic and cultural characteristics and prompt it to provide descriptions. Figures 20-22 demonstrate that all three models are capable of understanding the ethnic and cultural elements depicted in the images.

World memory. We investigate the models’ ability to recognize globally known celebrities, landmarks, logos, movies, food, plants, animals, and more. As illustrated in Figures 23-38, we observe that the models generally identify them correctly. However, when the images reflect information updated recently, the models’ descriptions are not accurate, as depicted in Figure 24. Additionally, when asked about information relevant to real people, GPT-4V occasionally opts to refrain from responding, as shown in Figure 29.

###### Sec. 2.1 Object-Centric Perception

###### Spatial Relation Recognition

Prompt

Prompt

Who is the person on Jordan's left-hand side?

Tell me the spatial relation between the tennis ball and the man.

[Figure 1]

[Figure 2]

[Figure 3]

Gemini Pro

[Figure 4]

###### Gemini Pro

The person on Jordan's left-hand side is Kobe Bryant.

The tennis ball is to the left of the man.

[Figure 5]

###### GPT-4V

[Figure 6]

###### GPT-4V

The person on Jordan's left-hand side, wearing the yellow Lakers jersey with the number 8, is Kobe Bryant.

The tennis ball is in front of the man and slightly to his left, indicating that he has either just hit the ball or is preparing to hit it. The ball is at a lower height, roughly around the waist level of the man, and it appears to be very close to his racket, suggesting a moment of action during a tennis match.

[Figure 7]

SPHINX

The person on Jordan's left-hand side is Michael Jordan.

[Figure 8]

SPHINX

The tennis ball is positioned below the man, as he is swinging his tennis racket to hit it.

- Figure 1: Results on spatial relation recognition. None of the three models can answer correctly. Red highlights the wrong answer. Yellow highlights the incompetence in performing the task.

Refer to Section 2.1 for detailed discussions.

###### Sec. 2.1 Object-Centric Perception

###### Spatial Relation Recognition

Prompt

In the first image, Curry's left-hand side is Green. In the second image, Wade's left-hand side is James. In the third image, who is the person on Jordan's left-hand side?

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

###### Gemini Pro

GPT-4V

Bryant.

The person on Jordan's left-hand side in the third image is LeBron James, wearing a Lakers number 23 jersey.

[Figure 14]

SPHINX

The person on Jordan's left-hand side is Kobe Bryant.

- Figure 2: Results on spatial relation recognition. By providing two examples, GPT-4V accurately identifies spatial relationships, whereas the other two models still respond incorrectly. Green

highlights the right answer. Red highlights the wrong answer. Refer to Section 2.1 for detailed discussions.

###### Sec. 2.1 Object-Centric Perception

###### Object Counting

###### Prompt

###### Prompt

Prompt

Count the number of fruits in the image.

Count the number of books in the image.

Count the number of black birds in the image.

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

###### Gemini Pro

###### Gemini Pro

[Figure 20]

###### Gemini Pro

There are 4 fruits in the image.

There are 8 black birds.

There are 5 books.

[Figure 21]

[Figure 22]

###### GPT-4V

###### GPT-4V

[Figure 23]

###### GPT-4V

There are eight black birds visible in the image.

There are three fruits in the image.

There are five books in the image.

[Figure 24]

[Figure 25]

###### SPHINX

###### SPHINX

[Figure 26]

###### SPHINX

There are eight black birds in the image.

There are four fruits in the image.

There are two books in the image.

###### Prompt

###### Prompt

Count the number of people in the image.

How many face up dice have a count of 4?

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

###### Gemini Pro

###### Gemini Pro

There are 41 people.

There are 2 dice with a count of 4.

[Figure 31]

[Figure 32]

GPT-4V

GPT-4V

There are 30 people depicted in the image.

In the image you've provided, there are two dice that are facing up with a count of 4.

[Figure 33]

###### SPHINX

[Figure 34]

SPHINX

There are 100 people in the image.

There are two face up dice with a count of 4.

Figure 3: Results on object counting. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 2.1 for detailed discussions.

###### Sec. 2.1 Object-Centric Perception

###### Difference Spotting

###### Prompt

Prompt

What is the difference between the left part of the picture and the right part?

What is the difference between the left part of the picture and the right part?

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Gemini Pro

###### Gemini Pro

The difference is the shape of the head and the tail.

The boy on the left is holding a basketball while the boy on the right is holding a soccer ball. The boy on the left is wearing a red shirt and blue shorts while the boy on the right is wearing a blue shirt and red shorts.

[Figure 39]

###### GPT-4V

The left part of the picture displays a series of

[Figure 40]

overlapping circles that form the outline of a creature resembling a fish or a dinosaur, with a spiky back and a simple face with eyes and mouth indicated by circles. The right part shows a similar outline with a few key differences:

GPT-4V

The image shows two cartoon characters that appear to be identical in design, with the primary difference being the sports they are associated with. The character on the left is dressed in a basketball jersey, holding a basketball, suggesting he is a basketball player. The character on the right is dressed in a soccer kit, holding a soccer ball, indicating he is a soccer player. The outfits and the sports equipment are the key differences between the two halves of the picture.

- 1. The back spikes are represented by rectangles or squares, rather than pointed shapes.
- 2. The face is more abstract, with a triangular shape suggesting an eye, and there is no clear depiction of a mouth.
- 3. The overall appearance of the right side is more geometric and less organic than the left side. The left seems to represent a more organic and natural form, while the right appears more stylized and geometric.

[Figure 41]

###### SPHINX

[Figure 42]

The difference between the left part of the picture and the right part is that the left part shows a boy holding a basketball, while the right part shows a boy holding a soccer ball.

###### SPHINX

The difference between the left part of the picture and the right part is that the left part has a black and white dinosaur, while the right part has a black and white fish.

- Figure 4: Results on difference spotting. Green highlights the right answer. Red highlights the

wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 2.1 for detailed discussions.

###### Sec. 2.1 Object-Centric Perception

###### Difference Spotting

Prompt

What is the difference between the left part of the picture and the right part? Could you find five differences?

[Figure 43]

[Figure 44]

###### Gemini Pro

- 1. The left picture has four objects, while the right picture only has three objects.
- 2. The left picture has a white wall in the background, while the right picture has a wooden table in the background.
- 3. The left picture has a black cup on the table, while the right picture has a white cup with a lemon on it.
- 4. The left picture has a tall thermos, while the right picture has a shorter thermos.
- 5. The left picture has a smaller thermos, while the right picture has a bigger thermos.

[Figure 45]

###### GPT-4V

In the image you've provided, there are two parts, each showing a slightly different scene. Let's identify

five differences between the left part of the picture and the right part:

- 1. Lid of the Thermos: On the left, the thermos is closed with its lid on. On the right, the lid is off and placed beside the thermos.
- 2. Inner Container: The left side doesn't show an inner container outside of the thermos, while the right side shows a metallic inner container placed in front of the thermos.
- 3. Lemon Slice: A lemon slice is present on the right, placed on top of the coaster that is on the thermos lid. There is no lemon slice on the left.
- 4. Coaster: There is a coaster under the thermos on the right, which is not present on the left.
- 5. Light Switch Position: The light switches on the wall in the background are in different positions. On the left, the two switches closest to the thermos are down, while on the right, one is up and the other is down.

[Figure 46]

###### SPHINX

The left part of the picture has a lemon on top of one of the thermos bottles, while the right part of the picture does not have any lemon on top of the thermos bottles.

- Figure 5: Results on difference spotting. In the two images, there are only three differences, yet we ask the model to identify five. Both Gemini and GPT-4V attempt to accommodate our request and provide five differences, but neither performs well. Red highlights the wrong answer. Refer to Section 2.1 for detailed discussions.

###### Sec. 2.1 Object-Centric Perception

###### Optical Illusion Recognition

###### Prompt

Prompt

Which pear has higher brightness?

Does the tree in the picture above look like the person in the picture below?

[Figure 47]

[Figure 48]

[Figure 49]

Gemini Pro

Both pears have the same brightness.

[Figure 50]

###### Gemini Pro

[Figure 51]

###### GPT-4V

No, the tree in the picture above does not look like the person in the picture below.

The image you've provided appears to show two pears with a striped pattern in the background. Brightness in an image can be affected by many factors, including the hue, saturation, and luminance of the colors used, as well as the presence of other elements and the overall lighting of the image.

The tree is a large, green tree with a thick trunk and many branches. It is standing on a hilltop and there is a river and mountains in the background. The person is a young man with dark hair and he is wearing a black shirt and pants. He is dancing on a stage. The two images are very different and do not resemble each other.

To determine which pear has a higher brightness, we would typically consider which one appears lighter or more luminous compared to the other. However, due to variations in display screens and personal perceptions, what appears brighter to one person may not appear the same to another.

[Figure 52]

###### GPT-4V

To provide an objective measure, I can analyze the image and calculate the average brightness of each pear. Let's do that now.

The tree in the upper image does bear a resemblance to the pose of the person in the lower image. The way the tree branches out at the top can be likened to the angle at which the person's arms are bent, while the trunk aligns with the person's body. The visual similarity is likely coincidental but makes for an interesting comparison between the natural shapes in our environment and human poses.

The pear on the right side has a higher average brightness value (approximately 148.52) compared to the pear on the left (approximately 127.06). Thus, based on the average pixel values, the pear on the right is brighter.

[Figure 53]

SPHINX

[Figure 54]

###### SPHINX

No, the tree in the picture above does not look like the person in the picture below.

The pear on the right has higher brightness.

- Figure 6: Results on optical illusion recognition. GPT-4V shows a human-like visual understanding of optical illusions. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 2.1 for detailed discussions.

###### Sec. 2.1 Object-Centric Perception

###### Optical Illusion Recognition

###### Prompt

Could this be true in the real world?

[Figure 55]

[Figure 56]

###### Gemini Pro

No, it's not possible. This is an optical illusion.

[Figure 57]

###### SPHINX

No, this is not possible in the real world. The image shows a group of white dice arranged in a triangle shape, which is an artistic or creative representation. In the real world, dice are typically used for games and do not form specific shapes like a triangle.

[Figure 58]

GPT-4V

The image you've uploaded appears to show dice arranged in a way that defies the normal expectations of gravity and balance, suggesting an improbable structure. Without additional context, such as knowledge of the materials used or the method of assembly, it's difficult to definitively say whether it's a real, physical construction or a clever manipulation, be it digital or a trick of perspective.

In the real world, the arrangement shown would be very challenging to achieve due to the dice needing to balance on very small contact points, which would be unstable without some form of adhesive or support. However, it's not impossible with the use of supports that are hidden from view, or by modifying the dice to connect more securely. It's also possible that the image is a result of digital manipulation.

If the intent is to create such a structure in reality, one would likely need to employ some form of support or modification to the dice. If this is an artistic endeavor, it would be a fascinating project that combines elements of sculpture and illusion.

Figure 7: Results on optical illusion recognition. Only Gemini recognizes that this image is an optical illusion. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 2.1 for detailed discussions.

###### Sec. 2.2 Scene-Level Perception

###### Scene Understanding from Image

Prompt

Describe this image in details.

[Figure 59]

[Figure 60]

###### Gemini Pro

The image shows a large shopping mall with a glass roof. There are many people in the mall, most of whom are wearing masks. There are also a number of stores. The decorations in the mall are red and gold, and there are a number of large red balloons. There is a stage with a red carpet in the center of the mall. There are people on the stage, and there is a large screen behind them. There are also a number of people standing around the stage.

[Figure 61]

###### GPT-4V

The image depicts the interior of a bustling multi-level shopping mall decorated for an event. There are numerous red balloons lining the escalators and railings across the multiple floors, creating a festive atmosphere. The mall features a variety of stores and eateries visible on different levels, with signage for brands like McDonald's clearly visible.The central area is occupied by an event space, with

a crowd of spectators gathered around. The event includes several brightly colored obstacle courses or activity stations branded with

names like "PUKY" and "MOOVnGO." These stations are arranged in a row and consist of ramps, blocks, and other equipment for what appears to be a children’s play area or competition.Throughout the mall, people are engaged in various activities; some are watching the event, while others are walking around, shopping, or dining. The architecture of the mall includes a glass ceiling that allows natural light to flood the interior. The overall impression is one of a lively, family-friendly environment with a special event drawing the attention of the visitors.

[Figure 62]

###### SPHINX

The image depicts a bustling mall filled with people, where a large group of people is gathered in the center of the building. The mall is decorated with red and white balloons, creating a festive atmosphere. There are several people scattered throughout the mall, some walking around and others standing in groups.In addition to the people, there are a few objects that can be seen in the mall. There are two handbags, one near the left side of the mall and the other closer to the center. There are also two backpacks, one located near the left side and the other closer to the center.The mall is filled with a lively

energy, as people enjoy their time shopping and socializing in the busy space.

Figure 8: Results on scene understanding from image. GPT-4V’s descriptions are notably more detailed and exhibit fewer instances of hallucination. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 2.2 for detailed discussions.

###### Scene Understanding from Image

###### Prompt

Describe this image in details.

[Figure 63]

[Figure 64]

###### Gemini Pro

The image is a fantasy landscape with a large city in the center. The city is built on a series of interconnected platforms that are suspended in the air by large balloons. The platforms are filled with buildings,中には家、ショップ、寺院などがあります。 The city is surrounded by a large wall, and there are several airships docked at the city's edge. The sky is filled with clouds, and there are several mountains in the background.The image is very detailed, and there are many interesting things to see. The artist has created a very unique and imaginative world.

[Figure 65]

###### GPT-4V

The image depicts a fantastical scene that appears to be from a story or a game setting. There's a vast landscape with a variety of structures that seem to merge traditional architectural elements with fantastical ones, suggesting a world where different historical or cultural influences coexist.On the right side of the image, there's a towering building that resembles a pagoda with multiple levels, balconies, and overhanging roofs. It's built on a rocky outcrop and has many windows and openings, with flags and banners fluttering in the wind. The building's design suggests a blend of medieval and Asian architectural styles.To the left and in the background, there are more structures built into what appears to be a mountainous region. There are bridges connecting different areas, and the buildings are made of wood and stone with various designs, some resembling watchtowers or fortresses.Above the landscape, there are two airships with large balloons and what seem to be sails or wings, suggesting they are powered by both air and wind. The design of the airships is reminiscent of steampunk aesthetics, with wooden hulls and intricate details.In the foreground, we see a character with red hair, wearing leather armor and a sword strapped to his back. He’s looking at a handheld device that could be a map or a magical artifact. His attire and stance suggest he is an adventurer or a warrior. Another character in the background is walking away, dressed similarly in armor and carrying a bow, indicating that this might be a world of adventurers or warriors.The ground is rocky with patches of grass, and there are various objects scattered around like a cart, barrels, and crates, which add to the lived-in feel of the scene. The color palette is warm, with soft golden hues, suggesting either dawn or dusk. The sky is filled with soft clouds, further enhancing the mystical ambiance of the image.Overall, the image is rich with details that evoke a sense of adventure and fantasy, suggesting a world where exploration and perhaps combat are common.

[Figure 66]

###### SPHINX

The image is a beautifully rendered fantasy scene featuring a man standing in front of a large, colorful castle. The man appears to be a warrior, and he is holding a sword in his hand. The castle is situated on a hill, and it is surrounded by a picturesque landscape.In the sky above the castle, there are two hot air balloons flying, adding to the whimsical atmosphere of the scene. Additionally, there are two boats visible in the water, one near the castle and the other further away. The overall composition of the image creates a sense of adventure and wonder.

Figure 9: Results on scene understanding from image. Possibly influenced by the Japanese style of architecture in the image, Gemini intersperses a phrase in Japanese within their English response.

Green highlights the right answer. Yellow highlights the incompetence in performing the task. Refer to Section 2.2 for detailed discussions.

###### Scene Understanding from Image

###### Prompt

What are the connections between the top and the down pictures?

[Figure 67]

[Figure 68]

###### Gemini Pro

The photos show the aftermath of a car accident. The car is a yellow BMW M4. The car is heavily damaged. The front of the car is smashed in. The windshield is shattered. The car is on the side of the road. There are police officers and other people standing around the car. The police officers are investigating the accident. The other people are watching. The accident happened in Hong Kong.

[Figure 69]

###### SPHINX

The top picture shows a group of police officers standing in the street, while the bottom picture shows a group of people gathered on the sidewalk. The connection between the two pictures is that the police officers are likely responding to an incident or situation that has caused the crowd to gather on the sidewalk. The presence of the police officers suggests that they are trying to maintain order and ensure the safety of the people in the area.

[Figure 70]

GPT-4V

The top and bottom images appear to be two different perspectives of the same scene. Here are the connections between them:

- 1.Car Crash Sit: Both images show a yellow sports car that has crashed into the side of a building, suggesting they are depicting the same incident.

- 2. Emergency Response: There are emergency responders present in both images. In the top picture, police officers can be seen, while the bottom picture shows what appears to be a group of onlookers or additional responders.

- 3. Time of Day: It seems to be nighttime in both images, as indicated by the artificial lighting and dark sky.

- 4. Debris and Damage: There is debris scattered on the road in both images, and the damage to the building and car is consistent, further indicating that these photos were taken at the same event.

- 5. Angle of View: The top photo is taken at street level, likely by someone standing nearby, while the bottom photo is from an elevated position, possibly from a building across the street, looking down at the scene.

- 6. Police Presence: In the top image, the focus is on the police officers who are standing near the crashed car. The bottom image shows the broader context of the incident, with more bystanders and possibly more officers or emergency service personnel.The combination of these elements strongly suggests that both photographs are of the same event from different viewpoints, capturing the aftermath of a vehicle accident that occurred at night, involving a yellow sports car, with police and other individuals present at the scene.

- Figure 10: Results on scene understanding from image. Only GPT-4V correctly identifies that these two images represent different perspectives of the same scene. Green highlights the right answer. Refer to Section 2.2 for detailed discussions.

###### Sec. 2.2 Scene-Level Perception

###### Scene Understanding from Video

Prompt

Please describe this scene according to these temporal images.

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

###### Gemini Pro

The scene is an office break room with a brown mosaic wall, gray blinds on the windows, and gray and blue chairs. There are two round tables with white tabletops. On one table are two laptops and a phone. On the other table is a box of tissues. There is a wooden cabinet with a white countertop. On the countertop are three potted plants and an electric kettle. There is a blue chair in front of the cabinet. There is a gray chair in the corner of the room. There is a hallway next to the break room with white walls and gray carpet.

###### GPT-4V

[Figure 75]

The scene depicted across the three images is of a modern, indoor space, likely within a commercial or office building. The area is fitted with a casual seating arrangement suitable for brief meetings or breaks. In the images, you can see:

- 1. A room with a neutral color palette, featuring a series of grey upholstered chairs arranged around small, round white tables. The flooring is a polished stone tile, reflecting the light from overhead fixtures. The walls are adorned with a mix of wood paneling and large, closed window blinds, which suggest an emphasis on privacy and light control.
- 2. The same room viewed from a different angle shows more of the wood-paneled wall, decorated with a mosaic tile backsplash adding a touch of color to the otherwise muted tones of the room. This decoration is behind a counter that houses potted plants, which introduce a natural element to the space. Additional seating options are visible, including a chair with a distinct blue fabric and a metal base, differing from the wooden-legged chairs seen in the first image.
- 3. A third perspective offers a view of the entryway to the room, marked by a wooden door on one side and an open walkway on the other, leading to what appears to be a corridor carpeted in dark grey. This suggests that the seating area serves as a transitional space between different sections of the building.

The lighting is consistent across all three images, with a combination of recessed ceiling lights and natural light, which might be entering through the covered windows. Overall, the setting appears to be designed for functionality and comfort, providing a space where individuals can sit for short durations, either to rest, wait, or engage in informal discussions.

[Figure 76]

###### SPHINX

The image showcases a spacious room with a variety of furniture and decorative elements. There are several chairs scattered throughout the room, with some positioned near tables. The tables are accompanied by vases, which are placed at various locations. The room also features a wall with a unique design, adding to its aesthetic appeal.

- Figure 11: Results on scene understanding from video. Gemini correctly merges the information from the first two frames to accurately describe that the scene contains two round tables and three potted plants. Green highlights the right answer. Refer to Section 2.2 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Commonsense

[Figure 77]

GPT-4V

Prompt

In the scene presented in the image, the key behavior to pay attention to is the prohibition of smoking. The "No Smoking" sign is prominently displayed, indicating that smoking is not allowed in that area. This is likely a public space or a section within a business establishment where smoking is banned due to health regulations, consideration for non-smokers, or fire safety concerns. If you are in such a place, you should refrain from smoking and also be mindful of other similar signs that may indicate additional rules or guidelines to follow.

What behaviors should I pay attention to in this scene in the picture?

[Figure 78]

[Figure 79]

SPHINX

In the scene, there is a "no smoking" sign displayed on a wooden table. This sign serves as a reminder to pay attention to the behavior of smoking in the area. It is important to respect the rules and regulations regarding smoking in public spaces, as it can affect the comfort and well-being of others. If you are a smoker, you should ensure that you do not smoke within the vicinity of the sign or where others might be affected by your smoking.

[Figure 80]

###### Gemini Pro

The picture shows a "No Smoking" sign on a table. Therefore, the behavior to pay attention to is people smoking. If you see someone smoking, you should politely remind them that smoking is not allowed in the area.

Figure 12: Results on commonsense perception. Green highlights the right answer. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Commonsense

Prompt

[Figure 81]

###### Gemini Pro

What behaviors should I pay attention to in this scene in the picture?

The picture shows a library. In the library, people are reading, which is appropriate behavior. However, some behaviors are not appropriate in a library. For example, talking loudly, eating, and drinking are not allowed in most libraries. It is important to be respectful of others when using the library.

[Figure 82]

[Figure 83]

GPT-4V

In the picture, you see a library setting with individuals engaged in various activities. Here are some behaviors and aspects you might pay attention to:

Studying and Reading: Notice how individuals are focused on their reading materials or laptops, which suggests a quiet, studious atmosphere.

Seating Posture: Observe how people are seated. Good posture can reduce fatigue and prevent strain during long study sessions. Personal Space: The arrangement of tables and

[Figure 84]

SPHINX

chairs allows for personal space, which is important in public settings for comfort and concentration. Quiet Environment: Libraries are typically quiet zones. The calmness in the picture suggests that it's a place where silence is respected to maintain a conducive environment for studying. Use of Technology: Some individuals may be using laptops or other electronic devices, which indicates the integration of digital tools in learning and research. Organization: The neat arrangement of books and the clean environment suggest organizational standards in the library, aiding in the easy location of materials and maintenance of a pleasant environment. Hydration: You can see water bottles on the tables, which is a good practice to stay hydrated during long periods of study. Personal Belongings: Personal belongings are kept close to the individuals, which is important for security and easy access. This scene encapsulates the typical activities and behaviors expected in a library setting.

In the scene, there are several people sitting at tables and working on laptops in a library. As an AI, I cannot directly observe the behavior of the individuals in the image, but I can provide some general guidelines for library etiquette.

Be respectful of others: Keep your voice down and avoid loud conversations or phone calls. Keep noise levels to a minimum: Avoid playing music or using headphones at a high volume.

Keep the library clean: Dispose of any trash in designated bins and return books to their shelves after use.

Be mindful of personal belongings: Keep your belongings organized and avoid leaving them unattended. Follow library rules and policies: Adhere to any specific rules or guidelines set by the library, such as time limits for using the computers or quiet areas. By following these guidelines, you can help maintain a pleasant and productive environment for everyone using the library.

Figure 13: Results on commonsense perception. Green highlights the right answer. Yellow highlights the incompetence in performing the task. Refer to Section 2.3 for detailed discussions.

If I want to go to Antarctica, should I wear the clothes in Figure 1 or Figure 2? Please state the reason.

[Figure 85]

[Figure 86]

[Figure 87]

###### Gemini Pro

You should wear the clothes in Figure 2. Antarctica is the coldest continent on Earth, with average temperatures below freezing all year round. The clothes in Figure 1 are not suitable for such cold weather, as they are lightweight and do not provide enough insulation. The clothes in Figure 2, on the other hand, are much warmer and would be more appropriate for a trip to Antarctica. They are made of a thicker material and have a hood, which will help to keep you warm in the cold weather.

[Figure 88]

###### GPT-4V

For a trip to Antarctica, you should wear the clothes shown in Figure 2. The reason is that Antarctica is an extremely cold environment, with temperatures that can drop well below freezing. The clothing in Figure 2 appears to be a heavy, insulated parka with a fur-lined hood, which is designed to provide warmth in such cold conditions. This type of outerwear is essential for protection against hypothermia and frostbite.

On the other hand, the clothing in Figure 1 is a basic short-sleeve t-shirt, which offers very little insulation and is not suitable for cold climates. Wearing such a garment in Antarctica would not provide the necessary protection against the harsh elements and could be dangerous to your health due to the extreme cold.

[Figure 89]

###### SPHINX

You should wear the clothes in Figure 1. Antarctica is a cold and windy environment, so you would need warm and insulated clothing to protect yourself from the elements. The clothes in Figure 1 are designed to keep you warm and comfortable in cold weather, while the clothes in Figure 2 are not suitable for such conditions.

Figure 14: Results on commonsense perception. While Sphinx acknowledges the necessity of heavy garments in chilly conditions, it mistakenly classifies a T-shirt as appropriate for cold weather protection. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 2.3 for detailed discussions.

May I touch the object in these two pictures with my hands? Please analyze them separately.

[Figure 90]

[Figure 91]

[Figure 92]

###### Gemini Pro

The first picture shows a gas stove with a flame. You should not touch it with your hands because it is very hot and could burn you.

The second picture shows a fake flame that is powered by electricity. It is safe to touch with your hands because it does not produce any heat.

[Figure 93]

###### GPT-4V

The images you've provided show two different sources of heat. For the first image, it depicts an active gas stove burner with a blue flame. It is unsafe to touch the flames or the burner as it can cause severe burns. The stove is designed for cooking and should only be operated with proper kitchen utensils and safety precautions. The second image shows what appears to be an artificial decorative flame, typically used for ambiance. These are often electric and use lights, such as LEDs, and fabric to simulate the appearance of fire. While it's not as dangerous as real fire, the electric components and any heat they generate can still pose a risk. It's not advisable to touch the artificial flames, especially if the device is plugged in, to avoid the risk of electrical shock or burns from any heated elements. In summary, it's not safe to touch the objects in either picture with your hands due to the risk of burns or electrical shock.

[Figure 94]

###### SPHINX

The first image shows a gas stove with a flame on it, while the second image shows a hanging fire pit with flames inside.

Figure 15: Results on commonsense perception. Both Gemini and GPT-4V identify that the second image depicts flames disguised using lighting, but the distinction lies in Gemini’s inference that the objects in the image can be touched, whereas GPT-4V additionally warns of the risk of electric shock. Green highlights the right answer. Red highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Scientific Knowledge

###### Prompt

Prompt

What is the landform in the picture?

Which country is the finger pointing to in the image? What continent does this country belong to?

[Figure 95]

[Figure 96]

[Figure 97]

Gemini Pro

[Figure 98]

Gemini Pro

The landform in the picture is a desert.

India. Asia.

[Figure 99]

###### GPT-4V

[Figure 100]

The landform in the picture appears to be a desert landscape featuring large, wind-eroded rock formations known as yardangs. These are streamlined protuberances carved from bedrock or any consolidated or semi-consolidated material by the dual action of wind abrasion by dust and sand, and deflation, which is the removal of loose material by wind turbulence. Yardangs are typically found in desert environments where the wind direction is uniform and strong, and they are oriented with their long axis parallel to the prevailing wind direction, indicating the direction of the erosive winds. The smooth, rounded surfaces between the

GPT-4V

The finger in the image is pointing to India. India is located on the continent of Asia.

[Figure 101]

SPHINX

yardangs suggest the presence of sand dunes, which are

The finger is pointing to Australia, which is a country in the continent of Oceania.

formed and shaped by the wind as well. This combination of features suggests an arid, wind-swept environment with limited vegetation and moisture.

[Figure 102]

SPHINX

The landform in the picture is a group of large sand dunes.

Figure 16: Results on scientific knowledge perception. Green highlights the right answer. Red highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Scientific Knowledge

###### Prompt

Prompt

What physical phenomenon is reflected in the image?

What chemical experiment equipment is in the image?

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Gemini Pro

###### Gemini Pro

The image shows the physical phenomenon of refraction. Refraction is the bending of light as it passes from one medium to another. In this case, the light is passing from air to glass and then back to air. The glass has a higher refractive index than air, so the light bends more when it enters the glass and then bends back when it exits the glass. This causes the light to appear to be displaced when viewed from the side.

- 1. 2 glass beakers
- 2. 1 glass stirring rod
- 3. 1 large glass bowl

[Figure 107]

GPT-4V

The image shows a person pouring a liquid into a large beaker using a smaller beaker. This setup is common in chemical experiments where liquids are measured, transferred, or mixed. The larger container is a beaker, which is typically used in a laboratory setting for stirring, mixing, and heating liquids. It appears to have a spout for pouring. The smaller container looks like another beaker that is being used to transfer liquid into the larger one. There are no other distinctive features that suggest any other specific equipment or the nature of the liquid involved.

[Figure 108]

###### GPT-4V

The image shows a demonstration of light refraction and dispersion. It appears to feature two curved pieces of transparent material, likely acrylic or glass, with light beams

passing through them. The light beams are bending as they

pass through these materials, a process known as refraction. Additionally, the beams are spreading out into a spectrum of colors, which is indicative of dispersion — the separation of light into colors of different wavelengths due to varying degrees of refraction for each color. This is similar to the phenomenon that occurs in a prism, where white light is split into its constituent colors.

[Figure 109]

###### SPHINX

[Figure 110]

The image shows the phenomenon of light passing through two glass objects, creating a green and red light effect.

###### SPHINX

The image shows a beaker, a test tube, and a glass.

- Figure 17: Results on scientific knowledge perception. Green highlights the right answer. Red

highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Scientific Knowledge

###### Prompt

Prompt

What physical phenomenon is reflected in the image?

Why can I still see my hand in the mirror when the book covers the hand?

[Figure 111]

[Figure 112]

[Figure 113]

Gemini Pro

[Figure 114]

###### Gemini Pro

The image shows the dispersion of light. This is a phenomenon that occurs when light passes through a prism or other transparent object and is broken up into its component colors. The different colors of light are refracted (bent) at different angles, causing them to spread out into a spectrum.

The book is not covering your hand. The image in the mirror is flipped, so it looks like the book is covering your hand, but it is actually covering the mirror.

[Figure 115]

[Figure 116]

GPT-4V

###### GPT-4V

The phenomenon you're observing in the image is due to the way light reflects off the mirror. When the light from your hand hits the mirror, it reflects back at the same angle it hits the surface. This is known as the law of reflection. Even though the book is covering your hand from direct view, the light reflecting off your hand is not blocked by the book, since the reflective angle is different from the angle at which you are looking at the hand. Therefore, you can still see the image of your hand in the mirror. This can create an interesting and sometimes confusing visual effect, as demonstrated in the photo you've shared.

The image shows a display of optical phenomena where light is interacting with transparent, prism-like objects. This interaction is causing the dispersion of light, which is a physical phenomenon where white light is separated into its component colors (typically seen as a spectrum ranging from red to violet). This occurs because different colors of light are refracted by different amounts when passing through a medium like a prism, due to their varying wavelengths. The result is the creation of a spectrum of colors, similar to what is seen in a rainbow. The sharp lines and patterns suggest that the light rays are being reflected and refracted by the objects, creating a visually striking array of colors on the surface beneath them.

[Figure 117]

###### SPHINX

[Figure 118]

###### SPHINX

You can still see your hand in the mirror because the mirror is reflecting the image of your hand, not the book. The book is placed on top of the mirror, and while it may partially obscure your view of the mirror, it does not block the reflection of your hand. The mirror's surface is smooth and reflective, allowing the image of your hand to be clearly visible through it.

The image reflects the physical phenomenon of light refraction, which occurs when light passes through a transparent object, such as a glass cube. In this case, the light is refracting through the glass cube, creating a visually striking effect with multiple colors and shapes. The refraction of light can also create rainbows, as seen in the image, where the light is bent and separated into its different wavelengths, producing a spectrum of colors.

- Figure 18: Results on scientific knowledge perception. Green highlights the right answer. Yellow highlights the incompetence in performing the task. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Historical Knowledge

Prompt

###### Prompt

What historical event is shown in the picture?

What historical event is shown in the picture?

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

Gemini Pro

###### Gemini Pro

The picture shows the Apollo 11 mission, which was the first manned mission to land on the Moon. The mission took place on July 20, 1969. The Apollo 11 crew consisted of Commander Neil Armstrong, Command Module Pilot Michael Collins, and Lunar Module Pilot Edwin "Buzz" Aldrin. Armstrong and Aldrin became the first humans to walk on the Moon. The Apollo 11 mission was a major milestone in the history of space exploration.

The picture shows the fifth Solvay Conference on Physics in 1927.

[Figure 123]

GPT-4V

The picture you have uploaded is of the Solvay Conference, specifically the fifth conference held in October 1927 in Brussels. It was one of the most famous and influential conferences in the history of physics, attended by some of the most notable figures in the field at the time. The conference was centered on the newly formulated quantum theory, and the discussions that took place helped shape the future of quantum mechanics. Notably, the photo captures key figures such as Albert Einstein, Niels Bohr, and Marie Curie among others.

[Figure 124]

###### GPT-4V

The picture depicts an astronaut on the lunar surface with the Lunar Module and an American flag, indicating a moment from the Apollo moon landings. This event is significant as it represents one of the human spaceflight missions conducted by NASA during the late 1960s and early 1970s, showcasing a landmark achievement in human space exploration.

[Figure 125]

SPHINX

The picture shows a group of men from the 1920s, which is a historical event.

[Figure 126]

SPHINX

The historical event shown in the picture is the Apollo 11 moon landing, where an astronaut is standing on the moon with an American flag.

Figure 19: Results on historical knowledge perception. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Multicultural Customs

[Figure 127]

###### Gemini Pro

###### Prompt

The image shows four people performing a lion dance in front of a traditional Chinese building. The performers are wearing colorful lion costumes and are moving in a coordinated fashion to the beat of a drum. The building in the background has a red roof and is decorated with red lanterns.

Provide a description of the image in English and also in the language native to the geolocation shown in the picture.

[Figure 128]

In Chinese:

图⽚显示了四个⼈在传统的中国建筑前表演舞 狮。表演者⾝穿五颜六⾊的狮⼦服装，随着⿎ 点に合わせて做着协调的动作。背景中的建筑 有红⾊的屋顶，并装饰有红⾊的灯笼。

[Figure 129]

###### GPT-4V

The image depicts a traditional Chinese lion dance. There are three colorful lion costumes, each operated by two dancers, performing on a red carpet. The lions are yellow, pink, and blue, adorned with intricate patterns and expressive faces. Behind them is a classical Chinese building with decorative red lanterns hanging from its eaves, which suggests that

[Figure 130]

SPHINX

In this image, there is a group of four colorful dragon costumes standing in front of a building. The dragons are arranged in a line, with one on the left, two in the middle, and one on the right. The costumes are vibrant and eye-catching, adding a festive atmosphere to the scene.

this scene may be set during a festive occasion, likely in a location with historical significance.

In Chinese, this description would be:

There are several people present in the image, with some standing near the dragon costumes and others scattered around the area. They appear to be observing the dragon costumes or participating in the event.

这张图⽚展示了中国传统的舞狮表演。有三套 ⾊彩鲜艳的狮⼦服装，每套都由两位舞者操作， 在红⾊的地毯上表演。狮⼦的颜⾊有⻩⾊、粉 ⾊和蓝⾊，装饰有复杂的图案和富有表情的⾯ 孔。他们背后是⼀座具有古典中国⻛格的建筑， 屋檐下挂着装饰性的红灯笼，这暗示这个场景 可能设置在⼀个有历史意义的地点，很可能是 在节⽇庆典期间。

The scene takes place in front of a building, which serves as a backdrop for the dragon costumes. The building has a red carpet laid out in front of it, adding to the overall festive ambiance of the scene.

Figure 20: Results on multicultural knowledge perception. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 2.3 for detailed discussions.

###### Multicultural Customs

###### Prompt

[Figure 131]

###### Gemini Pro

Provide a description of the image in English and also in the language native to the geolocation shown in the picture.

The image shows a group of Japanese women performing a traditional dance. They are wearing traditional Japanese clothing and are playing traditional Japanese instruments. The background is a green wall with gold flecks.

[Figure 132]

In the native language of the geolocation shown in the picture, the image would be described as follows:

画像には、⽇本の伝統舞踊を披露する⼥性た ちのグループが映っています。彼⼥たちは⽇ 本の伝統⾐装を着て、⽇本の伝統楽器を演奏 しています。背景は緑の壁で、⾦⾊の斑点が あります。

[Figure 133]

###### GPT-4V

The image depicts a traditional Japanese scene with four individuals. In the foreground, there are three people seated in a row, dressed in black kimonos, playing shamisen, a traditional Japanese three-stringed instrument. Each musician is focused on their instrument, which rests on a small wooden platform. The background features a performer in the center, standing and dressed in a vibrant, elaborate kimono with a bold design of waves and clouds in shades of purple, red, and white. The performer is wearing traditional white face makeup and a black wig styled in a classical manner, holding a paper umbrella and wearing traditional geta footwear. This setting suggests a cultural performance, likely related to Kabuki theatre.

In Japanese:

この画像には、伝統的な⽇本のシーンが描かれています。前景には、黒い着物を着た3⼈の⼈物が⼀列に 座り、三味線という⽇本の伝統的な三弦楽器を演奏しています。各ミュージシャンは、⼩さな⽊製の台の 上に置かれた楽器に集中しています。背景には、中央に⽴つパフォーマーがいて、波と雲の⼤胆なデザイ ンが施された紫、⾚、⽩の鮮やかな⾊彩の豪華な着物を⾝にまとっています。このパフォーマーは、伝統 的な⽩い顔の化粧と古典的なスタイルの黒いかつらを着⽤し、和傘を持ち、伝統的な下駄を履いています。 この設定は、歌舞伎に関連した⽂化的なパフォーマンスを示唆しています。

[Figure 134]

###### SPHINX

In this image, there is a group of people dressed in traditional Japanese clothing, known as kimonos. They are playing musical instruments, with some of them holding umbrellas. The group consists of four people, with one person standing out from the rest. This person is holding an umbrella and appears to be a geisha, a professional female entertainer in Japan. The other three people are playing musical instruments, with two of them playing guitars and one playing a drum. The scene captures a lively and cultural moment in Japan.

Figure 21: Results on multicultural knowledge perception. Green highlights the right answer. Refer to Section 2.3 for detailed discussions.

###### Multicultural Customs

###### Prompt

[Figure 135]

Provide a description of the image in English and also in the language native to the geolocation shown in the picture.

###### Gemini Pro

The image shows a man and a woman wearing traditional Korean clothing, or hanbok, in front of a traditional Korean building. The man is wearing a black gat and the woman is wearing a pink hanbok with a floral pattern. The woman is holding a mirror and applying makeup while the man is looking at his phone.

[Figure 136]

이미지는 한복을 입은 남녀가 전통 한옥 앞에 서 있는 모습을 보여줍니다. 남자는 검은색 가죽을 쓰고 여자는 분홍색 꽃무늬 한복을 입고 있습니 다. 여자는 거울을 들고 화장을 하고 남자는 휴 대전화를 보고 있습니다.

[Figure 137]

###### GPT-4V

The image shows two people dressed in traditional Korean hanbok. The individual on the left appears to be male, wearing a blue hanbok with a black gat (a traditional Korean hat), and is holding a smartphone in his hand, possibly taking a picture or reading something on the screen. The individual on the right appears to be female, wearing a white hanbok with pink floral patterns, holding what looks like a traditional hand mirror and seems to be gazing at her reflection. They are outdoors, and in the background, there is traditional Korean architecture, possibly a palace, and other blurred figures that suggest a public place with visitors.

In Korean:

이미지에는 한복을 입은 두 사람이 나옵니다. 왼쪽에 있는 사람은 남성으로 보이며, 파란 한복에 검은색 갓을 쓰고 있고, 스마트폰을 들고 사진을 찍거나 화면을 보고 있는 것 같습니다. 오른쪽에 있는 사람은 여성으로 보이며, 핑크색 꽃무늬가 있는 흰색 한복을 입고 있고, 전통 손거울을 들고 있는 듯 거울에 비친 자신의 모습 을 바라보고 있습니다. 그들은 야외에 있고 배경에는 한국 전통 건축물, 아마도 궁궐이 보이며, 주변에 흐릿 하게 보이는 다른 인물들이 관광객이 있는 공공장소임을 암시합니다.

[Figure 138]

###### SPHINX

In the image, a man and a woman dressed in traditional Korean clothing are standing together. The woman is holding a cell phone, and the man is holding a mask. They appear to be taking a selfie together.

In the background, there are several other people walking around, possibly enjoying the outdoor setting. The scene captures a moment of cultural exchange and shared experience between the couple dressed in traditional Korean attire.

Figure 22: Results on multicultural knowledge perception. Green highlights the right answer. Refer to Section 2.3 for detailed discussions.

Prompt

Prompt

Who is the person in the image?

Who is the person in the image?

[Figure 139]

[Figure 140]

[Figure 141]

###### Gemini Pro

The person in the image is a fictional character named Doctor Strange. He is a superhero who appears in Marvel comic books. He is a surgeon who becomes a powerful sorcerer after a car accident ruins his hands. He uses his powers to protect the world from magical threats.

[Figure 142]

###### Gemini Pro

[Figure 143]

###### GPT-4V

Taylor Swift

The image is a blend of two fictional characters: Iron Man, known for his red and gold power armor, and Doctor Strange, known for his mastery of the mystical arts. The character in the image appears to combine elements of both, wearing Iron Man's armor and also showing Doctor Strange's Eye of Agamotto and mystical symbols, suggesting a creative crossover of their abilities and aesthetics.

[Figure 144]

###### GPT-4V

The person in the image is Taylor Swift, a wellknown American singer-songwriter. The image appears to be promotional material for "The Eras Tour".

[Figure 145]

SPHINX

[Figure 146]

###### SPHINX

The person in the image is Iron Man, a popular superhero from the Marvel Comics universe.

The person in the image is Taylor Swift.

Figure 23: Results on celebrity recognition and description. Sphinx mistakes Doctor Strange in an Iron Man suit for Iron Man himself. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 2.3 for detailed discussions.

Prompt

Who is shooting the ball in the picture?

[Figure 147]

[Figure 148]

###### GPT-4V

The image you've uploaded appears to be from a basketball game. However, I'm not able to identify the person shooting the ball due to the policy

against identifying real people in images. If you're

looking for information on a specific player or game, I can try to help with general questions or provide information about the sport of basketball.

[Figure 149]

[Figure 150]

###### Gemini Pro

SPHINX

The player shooting the ball is Keldon Johnson.

In the picture, a basketball player is shooting the ball.

Prompt

Who is this person? Which team does he play for in the picture?

[Figure 151]

###### GPT-4V

[Figure 152]

The person in the image is a soccer player, but I am unable to provide the identity of real people in images. The soccer kit he is wearing in the photo is pink with the XBT sponsor logo, which is associated with the Miami-based soccer team, Inter Miami CF. This team competes in Major League Soccer (MLS), the highest level of soccer in the United States and Canada.

[Figure 153]

SPHINX

[Figure 154]

Thank you for the information. It is important to note that the image only shows the soccer player wearing a pink jersey and does not provide any context about the specific team or league he is associated with.

###### Gemini Pro

This is Lionel Messi. He plays for Paris SaintGermain in the picture.

Figure 24: Results on celebrity recognition and description. When the images depict recently updated information, Gemini’s descriptions tend to be inaccurate. GPT-4V, due to considerations of privacy protection, avoids addressing questions related to the identification of celebrities. Green highlights the right answer. Red highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Landmark Recognition and Description

###### Prompt

Prompt

Describe the landmark in the image.

Describe the landmark in the image.

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

###### Gemini Pro

Gemini Pro

The Canton Tower, also known as Guangzhou Tower, is a 604-meter (1,982 ft) multi-purpose observation tower in the Haizhu District of Guangzhou, Guangdong, China.

The image shows the Louvre Museum in Paris, France. It is one of the world's most famous and largest museums. The museum is home to some of the most iconic works of art in the world.

[Figure 159]

[Figure 160]

GPT-4V

###### GPT-4V

The image shows a tall, slender tower illuminated with vibrant, multicolored lights against a night sky. The tower has a lattice structure and tapers towards a pointed antenna at the top. There is a prominent crescent moon design lit up on the tower, adding a distinctive feature to its facade. The tower appears to be a prominent landmark, possibly used for telecommunications or as an observation tower. Below the tower, there are illuminated buildings and a bridge spanning a body of water, with reflections visible on the surface. This setting suggests the tower is located in a developed urban area near a river or waterfront. The tower's design and lighting resemble the Canton Tower in Guangzhou, China, known for its colorful nighttime display.

The image features the Louvre Pyramid, a large glass and metal pyramid that serves as the main entrance to the Louvre Museum in Paris, France. It was designed by the architect I. M. Pei and was completed in 1989. The pyramid is a striking example of modern architecture amidst the classical architecture of the Louvre Palace. It is illuminated, casting a warm glow and reflecting on the wet ground, which suggests that the photo was taken at night or during the evening after rain. The lights within the pyramid highlight its intricate metal framework.

[Figure 161]

SPHINX

The landmark in the image is the Louvre Pyramid, which is a glass and metal structure located in front of the Louvre Museum in Paris, France.

[Figure 162]

SPHINX

The landmark in the image is a tall, colorful tower that is lit up at night. It is located near a bridge and a river, creating a beautiful and vibrant scene.

- Figure 25: Results on landmark recognition and description. Green highlights the right answer. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Landmark Recognition and Description

###### Prompt

Prompt

Describe the landmark in the image.

Describe the landmark in the image.

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

###### Gemini Pro

Gemini Pro

The image shows the Pantheon in Paris, France. It is a domed church that was originally built as a church dedicated to Saint Genevieve and was later converted into a secular mausoleum containing the remains of distinguished French citizens.

The image shows the city of Paris, France. The Eiffel Tower is visible in the distance, and the Avenue des Champs-Élysées is in the foreground. The Place de la Concorde is in the center of the image, with the Obelisk of Luxor in the center of the square. The Grand Palais and the Petit Palais are located on either side of the Place de la Concorde.

[Figure 167]

GPT-4V

[Figure 168]

###### GPT-4V

The landmark in the image is the Panthéon, located in the Latin Quarter in Paris, France. It's a neoclassical building that originally served as a church dedicated to St. Genevieve but now functions as a secular mausoleum containing the remains of distinguished French citizens. The façade is modeled after the Pantheon in Rome, and its dome is a significant feature of the Parisian skyline.

The image features a view of the Eiffel Tower in the distance, which is a landmark located in Paris, France. It is seen from a high vantage point overlooking city buildings. In the foreground, there's a roundabout with a large, illuminated Christmas tree, suggesting that the photo was taken during the holiday season. The streets are lit with warm lights, and there appears to be a bustle of activity, which is common in urban settings. The sky is dusky, indicating either dawn or dusk.

[Figure 169]

SPHINX

The landmark in the image is the Capitole building, which is a large and impressive building with a domed roof.

[Figure 170]

SPHINX

The landmark in the image is the Eiffel Tower, which is a famous monument in Paris, France.

- Figure 26: Results on landmark recognition and description. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Logo Recognition and Description

Prompt

Prompt

Describe the logo in the image in details.

Describe the logo in the image in details.

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

###### Gemini Pro

###### Gemini Pro

The logo is a basketball with the word "Wilson" on it. The word "Wilson" is in black and has a yellow outline. The basketball is orange and has a black and white stripe around the middle. The logo also has the words "NCAA Final Four Edition" on it. The words "NCAA Final Four Edition" are in black and are located on the bottom of the basketball.

The logo is a red and yellow "M" inside a yellow square. The words "McDonald's" are written in red and yellow below the square. The logo is enclosed in a white square with rounded corners. The background of the logo is white.

[Figure 175]

###### GPT-4V

[Figure 176]

GPT-4V

The image shows a close-up of a food packaging box with a logo printed on it. The logo consists of a stylized cartoon character that appears to be a combination of a clown and a chicken. The character's face is predominantly white with black outlines and features. It has large, wide eyes, with black pupils and thick eyelashes, and a red nose. There are three red tufts of hair or feathers on the top of the head, resembling a comb. The character is smiling, revealing a red mouth and tongue. To the side of the character's head, there's a smaller, round object with a face, possibly representing an egg, featuring simplistic eyes and a mouth. Above these characters, there is non-Latin script text in black, presumably a name or slogan related to the brand. Below the characters, prominently displayed, is the familiar McDonald's logo with its iconic golden arches and the word "McDonald's" written underneath. There is also a red letter 'S' in a circle, likely indicating the size or type of the product contained within the box. The overall design has a playful and friendly aesthetic.

The logo in the image is of the brand "Wilson," known for producing sports equipment. The logo consists of the brand name "Wilson" written in a stylized, bold font. The letters are capitalized and have a slight italicization, suggesting motion and dynamism. The color of the logo appears to be a contrasting shade to the color of the basketball, providing clear visibility against the textured background of the ball. The basketball itself is labeled with "COMPOSITE LEATHER," indicating the material it is made from, and "NCAA," which stands for the National Collegiate Athletic Association, suggesting that this basketball meets the specifications for NCAA games. Additionally, there is a mention of "FINAL FOUR EDITION," which could imply that this basketball is a special edition for the Final Four, a term used to describe the last four teams remaining in the NCAA basketball tournament.

[Figure 177]

SPHINX

[Figure 178]

SPHINX

The logo in the image is for a basketball, specifically a Wilson NCAA Final Four Edition basketball. The logo is prominently displayed on the surface of the basketball, indicating its association with the NCAA (National Collegiate Athletic Association) and the Final Four, a prestigious basketball tournament for college teams.

The logo in the image is for McDonald's, a fast-food restaurant chain. It features the McDonald's arches, which are the company's iconic symbol. The arches are white and red, with the golden arches forming an "M" shape. The design is simple yet recognizable, and it is often associated with the brand's quick-service restaurants and the food they serve.

Figure 27: Results on logo recognition and description. Green highlights the right answer. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Logo Recognition and Description

###### Prompt

###### Prompt

Prompt

Describe the logo in the image in details.

Describe the logo in the image in details.

Describe the logo in the image in details.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

###### Gemini Pro

Gemini Pro

Gemini Pro

The logo is a black cat sleeping on the hood of a white car. The cat is facing the

The logo is a circle with a stylized "C" inside it. The "C" is made up of two curved lines that intersect in the middle. The logo is simple and elegant, and it conveys a sense of creativity and innovation. The colors of the logo are black and white, which are often associated with sophistication and elegance.

The logo is a wordmark of the band's name, "Linkin Park". The letters are in a bold, sans-serif typeface and are arranged in a horizontal line. The colors are black and white.

left of the image and its eyes are closed. The car is a Toyota Corolla and the logo is in the center of the hood.

[Figure 185]

[Figure 186]

GPT-4V

###### GPT-4V

The logo in the image is of Toyota, a well-known automotive manufacturer. It consists of three overlapping ellipses with a stylized "T" in the center. The central interlocking ovals represent the unification of the hearts of Toyota's customers and the company's products, while the larger oval surrounding these represents the global expansion of Toyota's technology and unlimited potential for the future. The background is typically white, and the ovals are in a metallic silver or chrome finish. This logo is prominently displayed on the front grille of Toyota vehicles.

The image shows a white baseball cap with a logo that consists of two words. The first word appears to be "LINKIN" and the second one is "PARK". The text is in capital letters, with a stencil or military-style font that features breaks or gaps within the strokes of the letters, suggesting a rugged or industrial aesthetic. The logo is likely associated with the music band of the same name, known for their nu metal and alternative rock style. The color of the text seems to be a shade of grey, which provides a soft contrast against the white background of the cap.

[Figure 187]

###### GPT-4V

The logo in the image appears to be a stylized knot. It consists of intertwined lines forming a continuous loop with no visible beginning or end. The design is simple, elegant, and appears to be monochromatic, based on the grayscale image. The knot is symmetrical and may represent connectivity, complexity, or infinity due to its looping nature. The background is plain, ensuring that the logo stands out prominently. The overall design conveys a sense of sophistication and modernity.

[Figure 188]

SPHINX

[Figure 189]

SPHINX

The logo in the image is for the band Linkin Park. It is a white baseball cap with a black logo on the front. The cap is placed on a black and white surface, which could be a chair or a table.

[Figure 190]

SPHINX

The logo in the image is for Toyota.

The logo in the image is a black and white circle with a black line in the middle.

- Figure 28: Results on logo recognition and description. Green highlights the right answer. Red

highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Movie Recognition and Description

###### Prompt

Please name three movies that this actor has starred in.

[Figure 191]

GPT-4V

[Figure 192]

The actor in the image is known for a wide range of films, particularly in the action genre. Three movies that he has starred in are:

- 1. "The Transporter" - This film is where he plays a skilled driver and mercenary who will deliver anything, anywhere – no questions asked, for the right price.
- 2. "Crank" - In this high-octane action film, he plays a hitman who must keep his adrenaline levels elevated to keep himself alive after being poisoned.
- 3. "Snatch" - A British crime comedy film where he takes on the role of an unlicensed boxing promoter who finds himself in the midst of various intertwined plots involving a stolen diamond.

[Figure 193]

Gemini Pro

[Figure 194]

SPHINX

Jason Statham has starred in the following movies:

- 1. The Expendables
- 2. The Transporter
- 3. The Italian Job

Jason Statham has starred in several movies, including "The Transporter," "Crank," "The Expendables," "Transporter 2," and "The Mechanic."

Prompt

[Figure 195]

###### Gemini Pro

Please name three movies that this actor has starred in.

This is the actor Wu Jing. He has starred in movies such as Wolf Warrior 2, The Wandering Earth, and The Battle at Lake Changjin.

[Figure 196]

[Figure 197]

###### GPT-4V

I'm sorry, but I can't help identify or make assumptions about people in images. If you have any other questions about cinema or actors, feel free to ask!

[Figure 198]

SPHINX

I'm sorry, but I cannot provide a list of movies that this actor has starred in as there is no information about the actor in the image.

- Figure 29: Results on movie recognition and description. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Movie Recognition and Description

[Figure 199]

###### GPT-4V

###### Prompt

Describe this poster and tell me something based on it.

The image appears to be a movie poster for "Dune: Part Two." It features a large, crescent-shaped dune with a smooth, shadowed surface on one side and a bright, glowing edge on the other, suggesting sunlight reflecting off the sand. The contrast creates a striking visual effect, emphasizing the scale and the harsh environment. On the bright side of the dune, tiny figures are depicted, likely representing characters from the movie, suggesting a journey or mission they are undertaking in a vast desert.

[Figure 200]

The "Dune" series is based on the science fiction novels by Frank Herbert. The first book, "Dune," is particularly renowned for its deep political, religious, and ecological themes, set against the backdrop of an interstellar society with complex hierarchies. The story primarily revolves around the desert planet of Arrakis, also known as Dune, which is the only source of the universe's most valuable substance, the spice melange. This substance is crucial for space travel, extending life, and enhancing mental abilities.

"Dune: Part Two" is presumably a continuation of the story, following the events of the first film adaptation of Herbert's series, directed by Denis Villeneuve. The film is expected to explore the rise of the protagonist, Paul Atreides, and his confrontation with the forces that seek to control Arrakis and its precious resources. The themes of power, ecology, human potential, and destiny are likely to be further developed in this sequel.

[Figure 201]

###### Gemini Pro

The poster for Dune: Part Two features a large sand dune with a group of people walking across it. The title of the movie is written in the top left corner, and the release date is written in the bottom right corner. The poster is very similar to the poster for the first movie, which was released in 2021. The first movie was a critical and commercial success, so it is likely that the second movie will be as well.

[Figure 202]

SPHINX

The poster features a group of people walking on a sand dune, with the sun casting a shadow on the dune. The text on the poster reads "Dune Part Two," indicating that it is a continuation of a story or series.

Figure 30: Results on movie recognition and description. Green highlights the right answer.

Yellow highlights the incompetence in performing the task. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Movie Recognition and Description

Prompt

Which movie's classic shots are these pictures referring to?

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

###### Gemini Pro

GPT-4V

###### SPHINX

The first picture is a reference to the movie "The Matrix". The second picture is a reference to the movie "Inception". The third picture is a reference to the movie "Pulp Fiction". The fourth picture is a reference to the movie "The Dark Knight".

The images you've uploaded are reimagined scenes from the 1999 science fiction action film "The Matrix," directed by the Wachowskis. They show the protagonist, Neo, played by Keanu Reeves, in his iconic bullet-dodging pose. This scene is well-known for its pioneering use of visual effects, especially the "bullet

The pictures are referring to the iconic dance moves from the movie "The Matrix." The blue-colored man is seen dancing in a dark room, and the blue-colored woman is seen dancing in a room with a blue light.

time" effect.

###### Prompt

[Figure 210]

[Figure 211]

###### Gemini Pro

###### GPT-4V

Who is the director of this movie? Who are the main actors in this movie?

The director of this movie is Kazuya Nomura. The main actors in this movie are:

The image you've uploaded appears to be promotional artwork for a movie or media project, featuring a range of characters that suggest a science fiction or cyberpunk genre. However, without any specific details, titles, or recognizable characters, I cannot identify the director or the main characters of the movie from the image

[Figure 212]

- * Hiroki Yasumoto as Akira
- * Mamoru Miyano as Ryo
- * Kenjiro Tsuda as Jack
- * Yuko Kaida as Elena
- * Yui Ishikawa as Saya
- * Rikiya Koyama as Dan
- * Takahiro Sakurai as Makoto
- * Junichi Suwabe as Isao
- * Ami Koshimizu as Rio
- * Maaya Sakamoto as Yuki

alone. If you have more information or

another way to identify the movie, please provide that, and I can assist you further.

[Figure 213]

SPHINX

The director of this movie is Kazuaki Nakamura, and the main actors are Kiko Mizuhara, Yoshiki Inoue, and Takashi Matsuyama.

- Figure 31: Results on movie recognition and description. The image at the bottom is a fantasy piece created using a diffusion-based model and does not refer to any real-life movie. Gemini and Sphinx are misled into providing incorrect answers, whereas GPT-4V indicates its lack of knowledge regarding the relevant information. Green highlights the right answer. Red highlights the wrong

answer. Yellow highlights the incompetence in performing the task. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Food Recognition and Description

###### Prompt

Prompt

What are these dishes?

What are this dish made of?

[Figure 214]

[Figure 215]

[Figure 216]

Gemini Pro

[Figure 217]

###### Gemini Pro

Congee, egg, and dan tat.

The dish is made of rice noodles, pork, and lettuce. The noodles are thin and white, and they are served in a clear broth. The pork is thinly sliced and cooked until it is tender. The lettuce is added to the dish at the end, and it provides a refreshing contrast to the other ingredients.

[Figure 218]

###### GPT-4V

The image shows a tray with two dishes commonly found in East Asian cuisine, particularly Chinese. On one side, there is a bowl of what appears to be rice porridge or congee, a traditional breakfast dish made by cooking rice in a large amount of water until it breaks down into a porridge-like consistency. Accompanying the porridge is a pair of chopsticks. On the other side of the tray, there is a whole cooked egg and a half of a preserved egg, often referred to as a "century egg" or "thousand-year-old egg." This is an egg that has been preserved in a mixture of clay, ash, quicklime, salt, and rice hulls for several weeks to several months. The yolk becomes a dark green to grey color, with a creamy consistency and strong flavor, while the white becomes a translucent brown jelly with little flavor. These eggs are often eaten with congee.

[Figure 219]

GPT-4V

The dish in the image appears to be a bowl of noodle soup. It contains the following ingredients:

- - Noodles: These look like they could be a type of wheatbased noodle, commonly used in East Asian cuisines.
- - Meat: There are ground meat pieces that resemble pork, a common ingredient in many noodle soups.

[Figure 220]

[Figure 221]

###### SPHINX

###### SPHINX

This dish is made of noodles, meat, and vegetables, including lettuce and green onions.

These dishes are a bowl of rice, an egg, and a plate with a piece of meat on a green tray.

###### Prompt

What are these dishes?

[Figure 222]

###### GPT-4V

[Figure 223]

The image shows two pastry dishes. On the left, there appears to be a variation of a French pastry known as a "choux à la crème," which is a puff pastry filled with cream. This particular one has a green-colored cream, which might suggest a flavor like pistachio or matcha, and it is garnished with what look like slices of pistachio on top.On the right, there is a tart topped with fresh raspberries. It's likely a raspberry tart, which typically consists of a crisp pastry crust filled with crème pâtissière (pastry cream) or another type of cream, and topped with fresh raspberries.Both pastries are labeled with a sign that reads "PAIN CHAUD," which could be the name of the bakery or patisserie where these pastries are sold. "Pain chaud" literally means "hot bread" in French, indicating freshly baked goods.

[Figure 224]

[Figure 225]

###### Gemini Pro

SPHINX

The dishes are a pistachio cream puff and a raspberry tart.

These dishes are a cake and a dessert, both featuring raspberries as a prominent ingredient.

- Figure 32: Results on food recognition and description. Green highlights the right answer. Red

###### Sec. 2.3 Knowledge-based Perception

###### Food Recognition and Description

###### Prompt

Prompt

What are these dishes?

Describe these dishes.

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

###### Gemini Pro

Gemini Pro

Two plates with burgers and fries. The burgers are on toasted buns with lettuce, tomato, onion, and cheese. The fries are in a metal cup.

Fish and chips is a classic British dish made of fried fish and chips. The fish is usually cod or haddock, and the chips are made from potatoes.

It is often served with mushy peas or tartar

sauce.

[Figure 230]

GPT-4V

[Figure 231]

###### GPT-4V

The image shows two dishes. In the foreground, there is a burger that appears to have a glossy brioche bun, a wellgrilled beef patty, lettuce, tomato, and what could be either mayonnaise or a light aioli. There's a wooden skewer holding the burger together. Behind it is a second burger that seems to have a similar brioche bun, with a filling that looks like grilled beef, caramelized onions, and perhaps a slice of cheese or a sauce. To the right, there are two small metal buckets filled with French fries, which look crispy and lightly salted. The dishes are presented on white plates, and the setting suggests a casual dining environment, likely a restaurant

The image shows a classic dish of fish and chips, a popular British food. It typically consists of a battered and fried fish fillet, often cod or haddock, served with a side of fries, known as "chips" in the UK. Accompaniments may include a slice of lemon and tartar sauce, as seen in the image, which are used to add flavor to the fish.

specializing in American or fast-casual cuisine.

[Figure 232]

[Figure 233]

###### SPHINX

###### SPHINX

The dishes on the table include fish and chips, a

The image features a dining table with a variety of dishes and food items. There are two hamburgers, one placed on the left side of the table and the other on the right side. Alongside the hamburgers, there are two cups of french fries, one located near the top right corner of the table and the other closer to the bottom right corner. The table is set with a knife and a fork, one placed near the top left corner and the other near the bottom left corner. The arrangement of the dishes and food items suggests a casual dining experience, with a focus on burgers and fries.

bowl of fries, and a lemon wedge.

Figure 33: Results on food recognition and description. Green highlights the right answer. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Plant Recognition

###### Prompt

###### Prompt

What breed is the plant in the picture?

What breed is the plant in the picture?

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

###### Gemini Pro

Gemini Pro

The plant in the picture is a spider mum. Spider mums are a type of chrysanthemum that is characterized by its long, thin petals that resemble spider legs. They come in a variety of colors, including white, pink, red, and yellow. Spider mums are popular flowers for bouquets and arrangements, and they are also used in dried flower arrangements.

The plant in the picture is a Delonix regia, also known as the flame tree or royal poinciana. It is a flowering tree native to Madagascar and is widely grown as an ornamental tree in tropical and subtropical regions around the world. The flame tree is known for its vibrant red flowers, which are typically produced in large clusters. The leaves of the flame tree are bipinnate, meaning they are twice-divided into leaflets. The flame tree can grow to a height of 30-40 feet and is a popular choice for landscaping due to its beautiful flowers and attractive foliage.

[Figure 238]

GPT-4V

The plant in the picture is a chrysanthemum, often referred to as a mum. The variety shown here appears to be one of the spider mums,

[Figure 239]

###### GPT-4V

characterized by their long, tubular petals that

can be curved or hooked at the ends, resembling spider legs. This specific breed is hard to determine without more context, as there are many cultivars that look quite similar, but it's definitely within the chrysanthemum family, which is known for its vast diversity in flower shapes, sizes, and colors.

The plant shown in the picture is a flowering tree commonly known as the Royal Poinciana, Flamboyant, Flame Tree, or by its scientific name, Delonix regia. It's known for its bright red or orange flowers and fern-like leaves. It is native to Madagascar and is widely planted in tropical regions around the world for its striking appearance.

[Figure 240]

###### SPHINX

[Figure 241]

SPHINX

The plant in the picture is a white-rayed mule's ears.

The plant in the picture is a Royal Poinciana, also known as Flamboyant.

- Figure 34: Results on plant recognition and description. Green highlights the right answer. Red

###### Sec. 2.3 Knowledge-based Perception

###### Animal and Plant Recognition

###### Prompt

###### Prompt

What breed is the plant in the picture?

What breed is the plant in the picture?

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

###### Gemini Pro

Gemini Pro

It is not a real plant. It is a Lego flower bouquet.

The plant in the picture is a tulip. It is a perennial, bulbous plant with showy flowers in

the genus Tulipa, of which about 75 wild species are currently accepted and which belongs to the family Liliaceae. The name 'tulip' is thought to be derived from a Persian word for turban, which it may have been thought to resemble.

[Figure 246]

GPT-4V

The item in the picture that appears to be a plant is not actually a living organism. It seems to be a creative decorative piece, possibly made from Lego or similar building blocks, designed to look like a colorful bouquet of flowers. Since it's not a real plant, there is no breed associated with it. These kinds of objects are often used for office or home decoration, and they can be a playful and low-maintenance alternative to real plants.

[Figure 247]

###### GPT-4V

The "plant" in the image you've provided is not a living plant, but rather a plush or fabric-based representation designed to resemble a cactus. It's made of green textile material, and the blue "flowers" appear to be made of a different fabric, possibly velvet or another soft material. This is a decorative item, not a real plant species.

[Figure 248]

###### SPHINX

The plant in the picture is a sunflower. The plant in the picture is a sunflower.

[Figure 249]

###### SPHINX

The plant in the picture is a cactus.

###### Prompt

[Figure 250]

What animal is the pen pointing to in the image? Gemini Pro

[Figure 251]

The pen is pointing to a rabbit.

[Figure 252]

GPT-4V

The pen in the image is pointing to a chicken.

[Figure 253]

SPHINX

The pen is pointing to a dog in the image.

- Figure 35: Results on animal and plant recognition and description. Green highlights the right

answer. Red highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Artwork Recognition and Description

###### Prompt

Prompt

Describe the artwork in the picture.

Describe the artwork in the picture.

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

###### Gemini Pro

Gemini Pro

The artwork in the picture is a sculpture of David by Michelangelo. The sculpture is made of marble and is located in the Galleria dell'Accademia in Florence, Italy. It was created between 1501 and 1504. The sculpture depicts David, a biblical figure, at the moment just before he fights Goliath. David is shown as a young man with a muscular build and a determined expression on his face. He is nude except for a loincloth, and his hair is short and curly. The sculpture is considered to be one of the masterpieces of Renaissance art and is one of the most famous sculptures in the world.

The artwork in the picture is a human figure made of clay. The figure is standing with its arms outstretched to the sides. The head is featureless, and the body is slightly hunched over. The figure is likely a representation of a human ancestor or spirit. It was found in a tomb in China and is believed to be over 6,000 years old.

[Figure 258]

###### GPT-4V

The image features an ancient terracotta sculpture displayed in a museum setting. The sculpture appears to be anthropomorphic, with a simplified human form featuring a distinct head, arms, and torso. The head has stylized facial features including eyes, a nose, and a mouth, giving it a somewhat abstract or expressionistic look. The arms are outstretched to either side, suggesting a pose that might indicate an offering or display.The sculpture seems to be aged, with a rough, uneven texture that speaks to the material it is made from and the wear it has endured over time. There's a visible coloration that might be due to the original pigments or the result of aging. It is presented on a simple, modern, dark grey or black pedestal, which contrasts with the lighter color of the terracotta and highlights the artifact.In the background, there are diagrams and photographs that provide additional context or information about the sculpture. However, the details of these background elements are not clear enough to discern specific information. The overall lighting and presentation suggest that this is an object of significance, possibly from an archaeological or historical exhibition.

[Figure 259]

GPT-4V

The artwork in the picture is a classical sculpture, most likely from the Renaissance period, judging by the style and technique. The figure is a male, depicted in a pensive pose, with one hand raised to his chin, suggesting a moment of deep thought or contemplation. The sculpture is highly detailed, with intricate curls in the hair and a lifelike representation of facial features. The level of detail in the hair and the musculature suggests a high level of skill on the part of the sculptor. The material appears to be marble, indicated by the smooth texture and the way light is diffused on the surface.It's a portrait bust, meaning it likely includes just the head and the upper part of the torso. This type of sculpture was commonly used to depict figures from mythology, historical individuals, or, more rarely, a conceptual figure representing an allegory or an idea. Without more context, it's not possible to determine the exact identity or the intended symbolism behind this particular sculpture.

[Figure 260]

[Figure 261]

###### SPHINX

###### SPHINX

The artwork in the picture is a sculpture of a

The artwork in the picture is a close-up of a statue, showcasing the intricate details of the sculpture. The statue appears to be a bust of a man, with a prominent nose and a thoughtful expression. The lighting in the image highlights the texture and depth of the statue, allowing viewers to appreciate the skill and craftsmanship that went into creating this piece of art.

person with outstretched arms, displayed on a

pedestal.

- Figure 36: Results on artwork recognition and description. Green highlights the right answer. Red

highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Artwork Recognition and Description

###### Prompt

Prompt

Describe the artwork in the picture.

Describe the artwork in the picture.

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

###### Gemini Pro

Gemini Pro

The artwork in the picture is a bronze vessel called a ding. It has four legs and a rectangular body with a lid. The body is decorated with taotie masks and geometric patterns. The ding is a type of ritual vessel that was used in ancient China for cooking and serving food. It is a valuable example of Chinese art and history.

The artwork in the picture is a gold mask with a human face. The mask is very realistic, with detailed features and even eyelashes. The mask is also very thin, and it is unclear how it was made. The mask is likely to be a ceremonial object, and it may have been used in religious rituals. The mask is from the Sanxingdui culture, which flourished in the Sichuan Basin in China from around 1200 to 600 BC.

[Figure 266]

GPT-4V

[Figure 267]

###### GPT-4V

The object in the image appears to be an ancient Chinese bronze vessel, known as a "Ding." It is a type of ritual bronzeware commonly found in Chinese tombs and used during the Shang and Zhou dynasties. This vessel stands on three or four legs and has two upward-facing handles. It is heavily ornamented with traditional Chinese motifs, including Taotie, which are stylized animal faces that are common in Chinese metalwork from this period. The patina on the vessel suggests considerable age, and the intricate designs indicate it was likely used by someone of high status, possibly in ritual ceremonies or as a burial item to accompany the deceased in the afterlife. The verdigris on the surface shows it has oxidized over time, which is typical for ancient bronze items.

The artwork in the picture appears to be a golden mask. It has a very stylized and abstract design, with prominent facial features that are exaggerated in size. The eyes are large and oval, and the mask includes representations of ears that extend out widely from the sides. The nose is prominent and protrudes outward, while the mouth area is less defined, giving the mask a somewhat enigmatic expression. The surface of the mask is textured, suggesting that it might have been crafted by hand, which adds a tactile quality to the piece. The mask has a sheen that indicates it might be made of gold leaf or a gold-colored metal, contributing to its striking visual impact. It's displayed against a dark background, which highlights its luminosity and golden hue.

[Figure 268]

###### SPHINX

[Figure 269]

###### SPHINX

The artwork in the picture is a bronze ware with intricate designs and patterns. It appears to be an old and valuable piece of art, possibly a bronze ware from the Shang Dynasty. The bronze ware has a greenish color and is displayed on a white background.

The artwork in the picture is a gold mask with a face design.

Figure 37: Results on artwork recognition and description. Green highlights the right answer.

Yellow highlights the incompetence in performing the task. Refer to Section 2.3 for detailed discussions.

###### Sec. 2.3 Knowledge-based Perception

###### Artwork Recognition and Description

Prompt

Describe the artwork in the picture.

[Figure 270]

[Figure 271]

###### Gemini Pro

A Riverside Scene at Qingming Festival, also known as Along the River During the Qingming Festival, is a Chinese

painting by the late Song-early Yuan dynasty artist Zhang Zeduan, created in the early 12th century. The handscroll

painting, one of the most famous works of Chinese art, depicts daily life along the Bian River in the capital city of Bianjing (modern-day Kaifeng, Henan) during the Qingming Festival.

[Figure 272]

###### GPT-4V

The image you've provided depicts a detailed, panoramic traditional East Asian scene, likely a Chinese landscape, executed in the style of a classical handscroll painting. It showcases a bustling river scene with a multitude of activities occurring both on the water

and along the riverbanks. There is a large, prominent arched bridge crossing the river, which is the central focal point of the artwork,

teeming with figures crossing over it.On the bridge and throughout the image, there are numerous people engaged in various activities, suggesting a lively community scene. The figures are rendered in fine detail, indicating their individual clothing and actions, which may represent different societal roles and daily tasks. The architecture of the buildings, including roof styles and the layout of the village, is indicative of historical East Asian designs.The color palette is muted, with earthy tones and subtle coloration, which is characteristic of many traditional East Asian paintings. The artwork likely represents a narrative or captures the essence of daily life during a particular historical period. The attention to detail in the depiction of the figures and the environment suggests that the artwork may have been created by a skilled artist with a deep understanding of the techniques and themes of traditional East Asian art.

[Figure 273]

###### SPHINX

The artwork in the picture is a painting of a bridge and a river, with a large crowd of people walking across the bridge. The painting is done in a style reminiscent of Chinese art, featuring a mix of brown and yellow colors. The people in the painting are of various sizes and positions, creating a lively and bustling atmosphere. The bridge itself is prominently displayed, spanning the width of the painting, while the river flows beneath it, adding to the overall scenic beauty of the image.

Figure 38: Results on artwork recognition and description. Green highlights the right answer.

Yellow highlights the incompetence in performing the task. Refer to Section 2.3 for detailed discussions.

###### 3 Advanced Cognition

On top of the fundamental perception, we further evaluate the more advanced cognition capabilities of Gemini, GPT-4V, and Sphinx. Cognition tasks for MLLMs require not only understanding multimodal concepts in visual contexts, but also conducting in-depth reasoning, problem-solving, and decision-making.

In Section 3.1, we first focus on the model’s ability to reason text-rich visual contents, including table and chart reasoning, along with visual code generation. In Section 3.3, we delve into MLLMs’ logical and quantitative comprehension to solve challenging science problems with pre-trained knowledge, e.g., mathematics and physics. In Section 3.2, our exploration targets on how the models reason abstract visual information from the tasks of abstract visual stimuli, Raven’s Progressive Matrices, and Wechsler Adult Intelligence Scale. In Section 3.4, we investigate the models’ understanding of emotions, through various scenarios such as facial expression analysis, image emotion analysis, and emotion-conditioned output. Finally in Section 3.5, we evaluate the decision-making performance of MLLMs in various intelligence games, including Sudoku and Go.

###### 3.1 Text-Rich Visual Reasoning

Table and chart reasoning. In Figures 39-40, we present two samples of flowchart understanding by the three models. As shown, Gemini can correctly summarize the high-level idea of flowcharts with brief sentences. GPT-4V tends to produce more detailed descriptions of the logical flow in the charts, but would occasionally make some mistakes. In contrast, Sphinx fails to extract the meaning of them, which is due to the lack of related pre-training data. In Figures 41-43, we evaluate the questionanswering performance on six different plots and tables. Similar to previous demonstrations, GPT-4V can respond with more reasoning details than Gemini. However, all three models have difficulties in providing a precise answer, which is mainly constrained by the unsatisfactory OCR accuracy. Also, as shown by the last sample, both Gemini and GPT-4V can understand the hand-drawing visual prompt, despite that Gemini provides the wrong final answers, indicating their generalization capacity for visual input.

Visual code generation. It’s an important skill for MLLMs to convert structured visual content into the corresponding codes. In Figures 44-45, we prompt the three models to generate LaTeX code of various mathematical formulas and render them for comparison. Overall, Gemini and GPT-4V exhibit better results than Sphinx, but still misrecognize some minor characters or symbols. Notably, for a rather complicated formula in printing form, both Gemini and GPT-4V generate correct codes. In Figures 46-47, we test the HTML code generation results for different types of websites. As shown, the HTML understanding capacity still exists a large improvement space for all three MLLMs. Only Gemini is capable of constructing the rough structure of simple websites, while GPT-4V simply identifies the text content. This might be also caused by the limited pre-training data.

###### 3.2 Abstract Visual Reasoning

Abstract visual stimuli. This task evaluates the visual abstract capabilities for object composition. As shown in Figures 48-49, GPT-4V exhibits the best abstract performance and also provides detailed descriptions for how the objects are composed of shapes. Instead, Gemini has partial abilities to recognize some simple abstract patterns, such as ‘boat’ and ‘house’, and Sphinx can not understand them.

Raven’s Progressive Matrices and Wechsler Adult Intelligence Scale. These two tasks are more challenging, since they require recognizing the high-level relations of different components, and predicting the next element in the matrices or sequences. As respectively shown in Figures 50-51 and 52-53, nearly all of the MLLMs are incorrect in the final answer. GPT-4V showcases some detailed reasoning process, but still struggles with the final prediction and can be easily misled by an incorrect intermediate step. This experiment indicates that, although the advanced MLLMs can initially identify the independent element, they fail to parse their relationship for further inference.

###### 3.3 Science Problem-Solving

Mathematical problems. Different from common visual question answering, the solving of mathematical problems involves both OCR capabilities from visual input and quantitative processing accuracy in the subsequent reasoning steps. In Figures 54-59, we show some mathematical problems concerning a wide range of tasks, including arithmetic, algebra, geometry, and integral calculus. The samples indicate that Gemini and GPT-4V can well tackle simple arithmetic and algebra problems. For more difficult trigonometry and integral calculus, they also exhibit favorable reasoning performance with the help of external tools. However, they are not very expert at recognizing the specific visual content in the images, such as numbers, symbols, and their correspondence. In addition, we observe that, with CoT techniques, i.e., “Please think step by step”, the previous wrong answer of Sphinx can rectified, demonstrating the importance of CoT prompting for open-sourced MLLMs.

Physics problems. Such problems further require MLLMs’ comprehension of the specialized vocabulary and concepts in Physics. In Figures 60-62, we show the problem-solving results of three MLLMs concerning dynamics, kinematics, and circuitry. As shown, Gemini and GPT-4V show wellperformed reasoning of Physics problems and well leverage the pre-trained specialized knowledge as reference. However, their performance can be limited by mathematical calculation, e.g., the range of integration, and the accuracy of physical equations, e.g., energy conservation equation. Due to the training data scarcity of Physics problems, the open-source Sphinx clearly lacks proficiency in solving such scientific problems with figures.

###### 3.4 Emotion Understanding

Facial expression analysis. In Figures 63-64, we evaluate the facial expression understanding capacity of different models. As shown, all of the three MLLMs exhibit good performance in this task. Therein, GPT-4V provides more dialectical thinking with rigorous analysis, e.g., the two possibilities of the first expression, while Gemini can directly respond with the accurate answer in a concise message. Also, GPT-4V and Sphinx both capture the truncated textual content on the plate of the third image, and incorporate this information into the reasoning. This result demonstrates their comprehensive visual understanding abilities.

Image emotion analysis. This task is more challenging, since there is no explicit facial expression shown in the image. Instead, MLLMs are required to indicate the implicit emotion conveyed from the visual concepts. As shown in Figures 65-69, we select diverse samples of various natural scenes and manufactured buildings. All three models can well depict the view first, and provide possible emotion within it. Therein, GPT-4V is observed to be neutral and emphasizes that emotions are subjective, and meanwhile gives a more comprehensive analysis. In contrast, Gemini tends to directly output the emotion preference, which corresponds with mainstream perspectives. In addition, Sphinx can achieve comparable performance to the other two MLLMs, indicating its superior emotion parsing capability.

Emotion-conditioned output. Different from predicting the emotion in the image, this emotionconditioned output enables MLLMs to describe the visual context conditioned on a pre-defined emotion, such as “in a romantic or terrifying way”. As shown in Figures 70-71, although Gemini and GPT-4V can correctly inject the corresponding emotion into the generated text, they both encountered hallucination issues, i.e., describing something that doesn’t exist, such as the “bike” and “shadows” in the first image, and the “sound” in the second image. This is might because of the too-powerful correlation abilities. In contrast, Sphinx is free from this issue, which exhibits the advantage of human-replicated MLLMs.

###### 3.5 Game Playing

Sudoku and Crossword. These two games are logic-based and combinatorial puzzles, evaluating MLLMs’ capabilities of OCR, structural understanding, and semantic reasoning. As shown in Figure 72 for Sudoku playing, if only taking an image as input, although Gemini tries to provide the answer within the output matrix, it cannot correctly identify the blank position, while GPT-4V and Sphinx fail to conduct the first-step OCR. Further, given the corresponding text input, both Gemini and GPT-4V can respond with the right answer. In Figures 73-74 for Crossword, the output of Sphinx is not very stable for the two examples, and Gemini performs slightly better than GPT-4V with more correct answers, which demonstrates its better structural comprehension and reasoning capacity.

Go playing. The game of Go involves intricate strategic thinking and rule-based decision-making, which has been regarded as a touchstone for general artificial intelligence. In Figures 75-76, we verify whether these three MLLMs obtain the ability to play go. As shown, they can master some initial skills, and predict the next move to a legal position in the chessboard. Although different strategies have different moving, only GPT-4V in the second sample obtains the optimal answer. However, Sphinx mistakenly views the game to be a Reversi in the third sample. This illustrates that current MLLMs still have a certain distance to cover before mastering advanced chess-playing capabilities.

###### Sec. 3.1 Text-Rich Visual Reasoning

###### Table and Chart Reasoning

Prompt

Please explain the flowchart.

[Figure 274]

[Figure 275]

###### Gemini Pro

The flowchart is a simple algorithm that compares two numbers, a and b. It starts by reading the values of a and b. Then, it checks if b is greater than a. If it is, then the flowchart displays b. Otherwise, it displays a. Finally, the flowchart ends.

[Figure 276]

###### GPT-4V

The flowchart represents a simple decision-making process with the following steps:

- 1.Start: This indicates the beginning of the process.
- 2.Read a, b: This step is about reading or inputting two values labeled a and b.
- 3.Is b > a?: This is a decision point where the process asks if the value of b is greater than a. ৹If the answer is YES, the flowchart instructs to:

ㆍDisplay B: Show or output the value of b. ৹ If the answer is NO, the flowchart instructs to: ㆍ Display A: Show or output the value of a.

- 4. End: This indicates the end of the process after either a or b is displayed, depending on the result of the comparison. In summary, this flowchart reads two values and displays the larger of the two. If b is greater than a, it displays b; otherwise, it displays a. The process terminates after displaying the value.

[Figure 277]

SPHINX

The flowchart is a visual representation of a decision-making process. It consists of a series of connected boxes, each representing a decision or a step in the process. The flowchart starts with the "Start" box, which is followed by the "YES" box. If the answer to the question is "YES," the flowchart proceeds to the "Is b=a" box, where the value of b is compared to the value of a. If b equals a, the flowchart moves to the "NO" box, indicating that the answer is no. If the answer is no, the flowchart ends at the "End" box. If the answer is yes, the flowchart proceeds to the "Display A" box, indicating that the value of A is displayed. If the answer is no, the flowchart ends at the "Display B" box, indicating that the value of B is displayed. The flowchart is designed to help decision-makers make choices based on the input values and the desired outcome. It is a useful tool for visualizing and organizing complex decision-making processes.

- Figure 39: Results on table and chart reasoning. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 3.1 for detailed discussions.

51

Note：The output of GPT-4V is the Markdown code and we show the rendered result.

###### Sec. 3.1 Text-Rich Visual Reasoning

###### Table and Chart Reasoning

Prompt

Please explain the flowchart.

[Figure 278]

[Figure 279]

###### Gemini Pro

The flowchart describes the process of reaching a consensus on a document. The process starts with a previous consensus, which is a document that has been agreed upon by all parties involved. If a user makes an edit to the document, the changes are implemented and the document is then edited further. If the document is not edited further, then a new consensus is reached and the process ends. However, if the document is edited further, then the process of seeking a compromise begins. This involves discussing the changes with the other parties involved and trying to reach an agreement on the changes. If an agreement is reached, then a new consensus is reached and the process ends. However, if an agreement is not reached, then the process of seeking a compromise continues until an agreement is reached.

[Figure 280]

###### GPT-4V

The flowchart appears to be outlining a process for editing and achieving consensus on an article, likely in a collaborative environment like a wiki or a shared document. Here's a step-by-step explanation:

- 1. Previous Consensus: The process begins with a previously established consensus on the content of the article.
- 2. Make an Edit: Someone makes an edit to the article.
- 3. Was the article edited further?: A decision point where one checks if there have been any further edits to the article after the initial one.

ㆍ If Yes, the process loops back to implementing the edit. ㆍ If No, the process moves forward to another decision point.

- 4. Do you agree with the changes?: This is another decision point where a person evaluates whether they agree with the changes made to the article.

ㆍ If they Agree, the process moves to establishing a New Consensus based on the changes. ㆍ If they Disagree, the process moves to seeking a compromise.

- 5. Seek the Compromise: Efforts are made to find a middle ground that satisfies all parties involved in the editing process.
- 6. Implement: Once a compromise is found, it is implemented in the article.
- 7. New Consensus: After successful implementation and agreement, a new consensus is reached regarding the content of the article. This cycle can repeat as necessary whenever new edits are made, ensuring continuous improvement and agreement among the collaborators.

[Figure 281]

SPHINX

The flowchart is a decision-making process that guides the editorial team in making decisions about an article. The process starts with the question "Was the article edited further?" If the answer is yes, the team proceeds to the next question, "Do you agree with the changes?" If the answer is no, the team moves on to the next question, "Disagree." If the answer is agree, the team proceeds to the next question, "Agree." If the answer is no, the team moves on to the next question, "New consensus." If the answer is yes, the team proceeds to the next question, "Implement." If the answer is no, the team moves on to the next question, "Seek the compromise." If the answer is yes, the team proceeds to the next question, "Make an edit." If the answer is no, the team moves on to the next question, "Previous consensus."

- Figure 40: Results on table and chart reasoning. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 3.1 for detailed discussions.

Note：The output of GPT-4V is the Markdown code and we show the rendered result.

###### Sec. 3.1 Text-Rich Visual Reasoning

###### Table and Chart Reasoning

[Figure 282]

Prompt

Approximately how many millions of RMB was the overall cloud computing market size in the region in 2020?

[Figure 283]

[Figure 284]

Gemini Pro

The answer is 2017. The paid penetration rate can be calculated by dividing the number of paid users by the total number of users. In 2017, the number of paid users was 300,000, and the total number of users was 700,000. Therefore, the paid penetration rate was 300,000 / 700,000 = 0.43, or 43%. This is the first time that the paid penetration rate reached 3% from 2013-2018.

[Figure 285]

###### Gemini Pro

[Figure 286]

###### GPT-4V

RMB 800 billion

To calculate the paid penetration rate for each year, we would need to follow this formula:

[Figure 287]

GPT-4V

[Figure 288]

We can extract the numbers from the image and perform the calculations for each year to find out which year first reached a 3% paid penetration rate. Let's start by analyzing the data from the image.

The bar chart you've provided shows the market size for private and public cloud in billions of RMB for the year 2020. To give you an accurate answer, I need to calculate the combined value of both private and public clouds for that year. From the visual estimation:

Now that we have the image displayed, we can visually inspect the data points on the graph. However, to precisely calculate the paid penetration rate for each year, we would need exact numerical values, which are not clear from this image alone.

- - The value for the private cloud seems to be approximately 400 billion RMB.

- - The value for the public cloud is slightly higher, around 600 billion RMB. Adding these values together: 400 (private cloud) + 600 (public cloud) = 1000 billion RMB Since 1 billion RMB is equal to 1000 million RMB, the overall market size in millions for 2020 would be: 1000 billion RMB * 1000 = 1,000,000 million RMB So, the overall cloud computing market size in the region in 2020 was approximately 1,000,000 million RMB.

We can attempt to estimate the values by using the image's scale, but for an accurate calculation, we would ideally have the actual data used to create this graph. Without precise figures, any calculation will be an approximation.

Let's attempt to estimate the values based on the graph's scale. The x-axis represents the years from 2013 to 2018. The y-axis has two scales: one for the user scale (in millions of dollars) and one for the paid user scale (in hundreds of thousands of dollars). We can approximate the values by judging their position on the y-axis relative to the known scale markers.

Once we have these approximated values, we can perform the calculations to estimate in which year the paid penetration rate first reached 3%. Let's proceed with this estimation.

[Figure 289]

###### SPHINX

Based on the estimated values from the graph, the paid penetration rate in China's digital music industry did not reach 3% in any of the years from 2013 to 2018. The closest it came to was approximately 2.5% in 2018. However, please note that these figures are based on visual estimations and the actual values may differ. For precise calculations, the actual numerical data is necessary.

The overall cloud computing market size in the region in 2020 was approximately 900.

###### Prompt

The paid penetration rate is derived by dividing the size of paid users by the total user size. So, which year was the first time China's digital music paid penetration rate reached 3% from 2013-2018?

[Figure 290]

SPHINX

2018

- Figure 41: Results on table and chart reasoning. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 3.1 for detailed discussions.

###### Sec. 3.1 Text-Rich Visual Reasoning

###### Table and Chart Reasoning

###### Prompt

I have outlined the items to be purchased and their quantities using a blue marker, please help me figure out how much DOLLAR I need to pay in total. Please think step by step.

Prompt

I want to buy two milk and one beer, please help me to figure out how much dollar I need to pay in total. Please think step by step.

[Figure 291]

[Figure 292]

[Figure 293]

Gemini Pro

[Figure 294]

###### Gemini Pro

The total cost of the items is $46.00. To calculate the total cost, we need to multiply the quantity of each item by its price and then add the results together. For example, the cost of 2 gallons of milk is 2 x $2.00 = $4.00. The total cost of all the items is $1.00 + $2.00 + $10.00 + $3.00 + $4.00 + $4.00 + $4.00 + $5.00 + $5.00 + $7.00 = $46.00.

The price of one milk is $2.00. The price of one beer is $7.00. So the total price is 2 * $2.00 + 1

* $7.00 = $11.00.

[Figure 295]

GPT-4V

The OCR has successfully extracted the text from the image. Now I will calculate the total cost for two milks and one beer based on the extracted prices. ㆍ The price of one milk is $2.00. ㆍ The price of one beer is $7.00. To calculate the total: Total cost = (Price of Milk x Quantity of Milk)

[Figure 296]

###### GPT-4V

The OCR has extracted the text, but it contains some errors and noise that we need to clean up. We can see the relevant items are 'Milk' and 'Beer' with their respective prices and quantities. Let's correct the OCR errors and calculate the total cost for these items. From the image, it appears that 'Milk' is $2.00 per unit and marked with 'x2' indicating two units, and 'Beer' is $7.00 per unit with no specified quantity, so we can assume it is one unit. I will now calculate the total cost. The total amount you need to pay for the items marked in blue (Milk and Beer) is $11.00.

+ (Price of Beer x Quantity of Beer) Let's do the math. The total amount you need to pay for two milks and one beer is $11.00.

[Figure 297]

###### SPHINX

The price of two milk is $2.00 x 2 = $4.00. The price of one beer is $7.00. So, the total price is $4.00 + $7.00 = $11.00.

[Figure 298]

###### SPHINX

The total cost of the items is $70.00.

- Figure 42: Results on table and chart reasoning. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 3.1 for detailed discussions.

###### Sec. 3.1 Text-Rich Visual Reasoning

###### Table and Chart Reasoning

###### Prompt

Prompt

Here are the order details for my taxi ride. Should I actually pay $29.42?

Here are the order details for my taxi ride. Should I actually pay $32.42?

[Figure 299]

[Figure 300]

Note:

Same query image as shown left, resized to save space.

[Figure 301]

Gemini Pro

No, you should not pay $32.42. The "Actually Paid" amount is the amount that you should pay. In this case, it is $29.42.

[Figure 302]

###### Gemini Pro

[Figure 303]

###### GPT-4V

No, you should not pay $29.42. The total fee is $32.42, but you have a $3 discount, so you should actually pay $29.42.

Based on the order details in the image you provided, the calculations for your taxi ride seem to have already been made for you. Here's a breakdown:

[Figure 304]

GPT-4V

- - Base Fare: $10
- - Mileage Fee: $12.11
- - Duration Fee: $10.32
- - Total before discount: $32.43 (It looks like there might be a rounding error here since $10 + $12.11 + $10.32 should be $32.43)
- - Discount: $3
- - Total after discount: $29.43 (Again, there seems to be a rounding error because if you subtract $3 from $32.43, it should be $29.43)

Based on the order details you've provided from your taxi ride:

- - Base Fare: $10
- - Mileage Fee: $12.11
- - Duration Fee: $10.32
- - Total: $32.42
- - Discount: $3
- - Actually Paid: $29.42

According to these details, you should actually pay $29.43, not $32.42. It seems that there's a discrepancy in the total and actual amount paid, likely due to a minor miscalculation. You may want to contact the service provider to clarify the final amount due.

The calculation seems correct. You start with a total of $32.42 for the combined fees. Then, a discount of $3 is applied. Subtracting the discount from the total gives you $32.42 - $3 = $29.42, which is the amount you should actually pay.

[Figure 305]

[Figure 306]

SPHINX

SPHINX

Yes, you should pay $29.42 for the taxi ride.

Yes, you should pay $32.42 for the taxi ride.

- Figure 43: Results on table and chart reasoning. Gemini provides a contradictory answer on the left part. Whereas, GPT-4V not only answers the question correctly but also identifies calculation errors in the bill on the right part. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 3.1 for detailed discussions.

###### Sec. 3.1 Text-Rich Visual Reasoning

Codes Generation

Prompt

Prompt

Please generate the latex code of the formula in the image.

Please generate the latex code of the formula in the image.

[Figure 307]

[Figure 308]

[Figure 309]

Gemini Pro

[Figure 310]

Gemini Pro

[Figure 311]

[Figure 312]

[Figure 313]

GPT-4V

[Figure 314]

GPT-4V

[Figure 315]

[Figure 316]

[Figure 317]

SPHINX

[Figure 318]

SPHINX

[Figure 319]

[Figure 320]

Figure 44: Results on visual code generation. Refer to Section 3.1 for detailed discussions.

56

Note：The output is the LaTeX code and we show the rendered result.

###### Sec. 3.1 Text-Rich Visual Reasoning

Codes Generation

Prompt

Prompt

Please generate the latex code of the formula in the image.

Please generate the latex code of the formula in the image.

[Figure 321]

[Figure 322]

[Figure 323]

Gemini Pro

[Figure 324]

Gemini Pro

[Figure 325]

[Figure 326]

[Figure 327]

GPT-4V

[Figure 328]

[Figure 329]

GPT-4V

[Figure 330]

[Figure 331]

SPHINX

(The result is none.)

[Figure 332]

SPHINX

[Figure 333]

Figure 45: Results on visual code generation. Refer to Section 3.1 for detailed discussions.

57

Note：The output is the LaTeX code and we show the rendered result.

###### Sec. 3.1 Text-Rich Visual Reasoning

Codes Generation

###### Prompt

Prompt

Generate the HTML code of the web page.

Generate the HTML code of the web page.

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

Gemini Pro

Gemini Pro

[Figure 338]

[Figure 339]

[Figure 340]

###### GPT-4V

[Figure 341]

GPT-4V

I'm unable to directly generate HTML code from an image. ……

I'm unable to directly generate HTML code from an image. However, if you have a specific section of the webpage you're interested in or need help understanding how to structure HTML for a similar layout, I can provide guidance on that. Could you specify which part of the web page you are interested in or what guidance you require?

[Figure 342]

###### SPHINX

1[0.97, 0.26, 0.98, 0.27].1[0.97, 0.27, 0.98, 0.28].1[0.97, 0.28, 0.98, 0.29].1[0.97, 0.29, 0.98,

- 0.3].1[0.97, 0.3, 0.98, 0.31].1[0.97, 0.31, 0.98, 0.32].1[0.97, 0.32, 0.98, 0.33].1[0.97, 0.33, 0.98, 0.34].1[0.97, 0.34, 0.98, 0.35].1[0.97, 0.35, 0.98, 0.36].1[0.97, 0.36, 0.98, 0.37].1[0.97, 0.37, 0.98, 0.38].1[0.97, 0.38, 0.98, 0.39].1[0.97, 0.39, 0.98,

- 0.4].1[0.97, 0.4, 0.98, 0.41].1[0.97, 0.41, 0.98, 0.42].1……

[Figure 343]

SPHINX

Model[0.21, 0.33, 0.27, 0.35].public. Here’s what to know.[0.21, 0.4, 0.37, 0.41].Gemini is Google’s largest and most capable AI that can process images, video, text, and even [0.21, 0.37, 0.78, 0.39].What is Google Gemini AI? How to Use the New Chatbot [0.21, 0.3, 0.74, 0.32].

NoteFigureprovided：Some46:byResultsGeminiof theonoutputresultsvisualiniscodea blankthegeneration.HTMLwebpage.codeForYellowandthe webpagehighlightswe showshownthetheincompetenceonrenderedthe right,inresult.theperformingresponse

the task. Red highlights the wrong answer. Refer to Section 3.1 for detailed discussions.

###### Sec. 3.1 Text-Rich Visual Reasoning

Codes Generation

Prompt

Prompt

Generate the HTML code of the web page.

Generate the HTML code of the web page.

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

Gemini Pro

Gemini Pro

[Figure 348]

[Figure 349]

[Figure 350]

GPT-4V

I'm sorry, but I can't assist with generating the HTML code of a webpage from an image. If you need help with HTML or have any other questions regarding web development or another topic, feel free to ask!

[Figure 351]

GPT-4V

[Figure 352]

[Figure 353]

SPHINX

19[0.73, 0.26, 0.75, 0.28].

[Figure 354]

SPHINX

<p>PHOTO</p>[0.48, 0.26, 0.53, 0.28].

FigureNote：47:SomeResultsofonthevisualoutputcodeisgeneration.the HTMLYellowcodehighlightsand wetheshowincompetencethe renderedin performingresult. the task. Red highlights the wrong answer. Refer to Section 3.1 for detailed discussions.

###### Sec. 3.2 Abstract Visual Reasoning

###### Abstract Visual Stimuli

###### Prompt

Prompt

The picture below shows a person made of tangram pieces. Please identify the movements of the person.

The tangram is a puzzle consisting of seven flat polygons that combine to form various shapes. What do the four shapes in the image look like?

[Figure 355]

[Figure 356]

[Figure 357]

###### Gemini Pro

[Figure 358]

Gemini Pro

- a. boat

- b. house

- c. windmill

- d. star

The correct answer is c. The person in a is missing an arm. The person in b has an extra piece. The person in d has the yellow piece in the wrong place.

[Figure 359]

GPT-4V

[Figure 360]

###### GPT-4V

The image you've uploaded appears to be a representation of Tangram solutions, which are made up of seven flat shapes called tans. Here's what each of the four shapes represents:

The image shows a series of four figures, each labeled a, b, c, and d, composed of tangram pieces. Each figure represents a different posture of a person, as follows:

- a. This shape resembles a boat with a tall sail on the left and the boat's hull at the bottom.

- b. This figure could be interpreted as a house with a pointed roof on top of a square base.

- c. The third shape looks like a larger triangle balanced on one tip, with smaller triangles and a square making up its interior.

- d. The final shape is more abstract, but it could be seen as a person or animal with outstretched limbs, or an X-like figure with a square in the center.

- Figure a: The person appears to be walking or stepping forward.

- Figure b: The person seems to be sitting with one leg extended.

- Figure c: The person looks to be standing with arms and legs spread out, similar to a jumping jack position.

- Figure d: The person might be depicted in a dynamic pose, possibly running or in mid-stride. Tangram puzzles like these allow for various configurations that can suggest movement or action, even though the figures themselves are static.

Tangrams are versatile and allow for a wide variety of patterns and figures to be created, often resembling animals, people, numbers, and various geometric arrangements.

[Figure 361]

[Figure 362]

SPHINX

###### SPHINX

I'm sorry, but I cannot see the image you are referring to. Can you please provide more information or a different image for me to analyze?

The four shapes in the image are colored triangles.

- Figure 48: Results on abstract visual stimuli. Green highlights the right answer. Red highlights

the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 3.1 for detailed discussions.

Illustration of GPT-4V’s ability to turn at highway ramp. Green highlights the right answer in understanding. Red highlights the wrong answer in understanding. Yellow highlights the incompetence in performing the task. Check Section 3.2 for detailed discussions.

60

###### Sec. 3.2 Abstract Visual Reasoning

###### Abstract Visual Stimuli

Putting it all together, the correct order of construction from first to last would be:

Prompt

- 1. (e) - Start with the base.
- 2. (c) - Add the first set of blocks on top of the base.
- 3. (d) - Add another layer, filling out the sides.

- 4. (a) - Further layers are added, with more details.
- 5. (b) - The final details are added, such as the crest on top.

The following five pictures show the different stages of building blocks, and below the pictures are their serial numbers. Please use the serial numbers to provide the correct order in which they were built.

[Figure 363]

[Figure 364]

SPHINX

1. 2. 3. 4. 5.

###### Prompt

Please analyze these images and provide the correct order.

[Figure 365]

[Figure 366]

###### Gemini Pro

The correct order is: a-c-e-d-b.

[Figure 367]

GPT-4V

[Figure 368]

To determine the correct order of the building blocks in the image, we should look for the progression of the construction. Here's the logical sequence based on what each image adds to the structure:

Gemini Pro

The correct order is a-b-c-d-e.

- 1. Start with the base or the most incomplete stage of the construction.
- 2. Look for the addition of new blocks on top of the existing ones.
- 3. Continue following the sequence where new blocks are added.
- 4. The final stage should be the most complete version of the construction. Let's analyze each labeled part:

[Figure 369]

###### GPT-4V

The drawing order of the bird based on the complexity and progression of the drawing from simple to more detailed would be:

- 1. Picture (a) - starting with a simple circle for the head.

- 2. Picture (d) - adding the outline of the body.

- 3. Picture (b) - drawing the beak and the eye.

- 4. Picture (c) - adding legs and a wing.

- 5. Picture (e) - final details are added, such as the pupil in the eye and the tail feathers. So, the order using the serial numbers would be: a, d, b, c, e.

- - (a) This part of the image has a substantial amount of the base constructed, with several layers on top.
- - (b) This looks more complete with additional blocks on top, including what seems to be the final piece which is the red block that acts as a crest.
- - (c) This has the base and a few layers but is less complete than (a).

- - (d) This is more complete than (c) but less than (a), as it lacks the upper green and red blocks present in (a).
- - (e) This is the most incomplete, with just the base layer of the structure.

[Figure 370]

SPHINX

(a), (c), (e), (b), (d).

- Figure 49: Results on abstract visual stimuli. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 3.2 for detailed discussions.

The results of different models for Temporal ordering. Green highlights the right answer in understanding. Red highlights the wrong answer in understanding.

###### Sec. 3.2 Abstract Visual Reasoning

###### Wechsler Adult Intelligence Scale

Prompt

Look the pattern on the left and select the missing item from the right. Please think step by step.

[Figure 371]

[Figure 372]

###### Gemini Pro

The triangle is divided into 3 triangles. The hexagon is divided into 6 triangles. So the missing item should be divided into 9 triangles. The answer is d.

[Figure 373]

###### GPT-4V

This image presents a logical sequence or pattern puzzle where one needs to determine the missing shape in the sequence based on the transformation applied to the first shape. The top row shows a transformation from a simple triangle to a more complex figure that retains the overall triangular shape but adds internal lines to create new shapes within it. The bottom row shows a hexagon followed by an arrow pointing to a question mark, suggesting that we need to determine the transformation applied to the triangle and then apply the same transformation to the hexagon to find the correct figure among the options provided. The transformation from the simple triangle to the complex figure appears to involve creating a smaller triangle inside the original one and then adding lines that connect the vertices of the inner triangle to the midpoints of the sides of the larger triangle. We should apply this same transformation to the hexagon: draw a smaller hexagon inside and then connect the vertices of the inner hexagon to the midpoints of the sides of the larger hexagon. Looking at the provided options for the hexagon:

- a. This figure does not have the midpoints of the larger hexagon connected to the smaller hexagon.
- b. This figure has additional lines that do not connect the vertices of the inner shape to the midpoints of the outer shape.
- c. This figure correctly connects the vertices of the inner hexagon to the midpoints of the outer hexagon's sides.
- d. This figure has too many lines, creating an overly complex shape that does not follow the transformation pattern.
- e. This figure does not depict a hexagon at all. Therefore, the correct answer would be option c, as it follows the transformation pattern applied to the triangle.

[Figure 374]

###### SPHINX

The missing item from the right is the letter 'a'. The pattern on the left is a triangle with a question mark. The pattern on the right is a hexagon with a letter 'a'.

- Figure 50: Results on Wechsler Adult Intelligence Scale. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 3.2 for detailed discussions.

Illustration of GPT-4V’s ability to turn at highway ramp. Green highlights the right answer in understanding. Red highlights the wrong answer in understanding. Yellow highlights the incompetence in performing the task. Check Section 3.2 for detailed discussions.

###### Sec. 3.2 Abstract Visual Reasoning

###### Wechsler Adult Intelligence Scale

Prompt

There are six letters a, b, c, d, e, and f written on the six sides of a cube. Based on the following four pictures, infer which letter is opposite to b? Please think step by step.

[Figure 375]

[Figure 376]

[Figure 377]

###### Gemini Pro GPT-4V

- In picture (1), e is opposite to a.
- In picture (2), b is opposite to c.
- In picture (3), f is opposite to e.
- In picture (4), d is opposite to a.

To determine which letter is opposite to 'b' on the cube, we need to analyze the four given pictures and see which sides of the cube are visible in each picture.

In the first picture, we see sides with letters 'a', 'e', and 'c'. In the second picture, we see 'a', 'b', and 'c'. In the third picture, we see 'e', 'f', and 'c'. In the fourth picture, we see 'e', 'd', and 'a'.

So we can infer that a is opposite to e, b is opposite to c, c is opposite to b, d is opposite to a, e is opposite to f, and f is opposite to e.

From these observations, we can make the following deductions:

- - 'a' is adjacent to 'e' and 'c' (from picture 1), and it's also adjacent to 'b' (from picture 2). Since a cube has six faces and 'a' is already shown to be adjacent to 'e', 'c', and 'b', it cannot be opposite to any of these.
- - 'c' is adjacent to 'a', 'e', and 'f' (from pictures 1 and 3), and also to 'b' (from picture 2), so it can't be opposite to 'b'.
- - 'e' is adjacent to 'a', 'c', and 'd' (from pictures 1 and 4), so it can't be opposite to 'b'.
- - Since 'b' appears only in the second picture and does not share a visible face with 'd' or 'f' in any of the pictures, we can deduce that 'b' is opposite to one of these two letters.

[Figure 378]

SPHINX

The four pictures show a cube with letters a, b, c, d, e, and f on the six sides. The letter opposite to b is c.

Now, because 'f' is adjacent to 'c' and 'e' (from picture 3) and we know 'c' and 'e' are adjacent to 'a' and 'b', 'f' cannot be opposite to 'b'. Therefore, by the process of elimination, 'd' must be the letter that is opposite to 'b' because it's the only letter that has not appeared on an adjacent side to 'b' in any of the four pictures.

- Figure 51: Results on Wechsler Adult Intelligence Scale. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 3.2 for detailed discussions.

###### Sec. 3.2 Abstract Visual Reasoning

###### Raven’s Progressive Matrices

Prompt

Among the eight options numbered 1 to 8 on the right side of the picture, which one is suitable to fill the missing position on the left side of the picture? Please think step by step.

[Figure 379]

[Figure 380]

###### Gemini Pro

For each row, the number of rings in the outer circle is one less than the sum of the rings in the inner circles. For example, in the first row, the outer circle has 1 ring, and the inner circles have 2 and 3 rings, and 2 + 3 = 5. In the second row, the outer circle has 2 rings, and the inner circles have 1 and 3 rings, and 1 + 3 = 4.Therefore, for the missing figure, the outer circle should have 3 rings, and the inner circles should have 2 and 4 rings. This corresponds to option 6.

[Figure 381]

###### GPT-4V

This puzzle appears to be a visual sequence or pattern recognition question, where you need to determine the rule that governs the sequence of shapes to find the missing piece. Let's analyze the given sequence:

- • In the first column, there are single circles.
- • In the second column, there are two concentric circles.
- • In the third column, there are three concentric circles. The number of concentric circles increases as we move down in each column. Also, the thickness of the circles varies. Let's apply this logic to find the missing piece:
- • The missing piece should be in the third column, which means it should have three concentric circles based on the pattern.
- • The thickness of the circles alternates in each row - thin, thick, thin. Since the second row has thick circles, the missing piece should have thin circles.

Now, looking at the options:

- • Options 1, 2, and 8 have three concentric circles.
- • Among these, option 2 has thick circles, so it doesn't fit the pattern.
- • Options 1 and 8 have thin circles, but option 8 has a solid center circle, which does not follow the pattern seen in the previous rows.

Based on this analysis, the suitable option to fill the missing position on the left side of the picture should be option 1, as it has three concentric circles with the correct thickness following the established pattern.

[Figure 382]

###### SPHINX

The missing position on the left side of the picture is marked with a question mark. The options 1 to 8 on the right side of the picture are numbered clockwise. The options 1 to 8 on the right side of the picture are marked with circles. The option that is marked with a circle and is clockwise from the question mark is 7.

Illustration of GPT-4V’s ability to turn at highway ramp. Green highlights the right answer in understanding. Red highlights the wrong answer in understanding. Yellow highlights the incompetence in performing the task. Check Section 3.2 for detailed discussions.

- Figure 52: Results on Raven’s Progressive Matrices. Red highlights the wrong answer. Refer to

- Section 3.2 for detailed discussions.

###### Sec. 3.2 Abstract Visual Reasoning

###### Raven’s Progressive Matrices

Prompt

Please look the pattern of the two rows and infer the two missing data. Please think step by step.

[Figure 383]

[Figure 384]

###### Gemini Pro

First row: 16, 15, 17, 14, _, _Second row: 32, 33, 31, 34, _, _The difference between the numbers in the first row is 1.The difference between the numbers in the second row is 2.The missing numbers are 13 and 30.

[Figure 385]

###### GPT-4V

Let's analyze the pattern in the two rows of numbers you've provided to determine the missing data. Here's the step-by-step thought process:

- 1. Identify the Pattern: We will look at the available numbers to identify any arithmetic or geometric sequences, common differences, or any other relationships between the numbers.
- 2. Apply the Pattern: Once we've identified the pattern, we will apply it to find the missing numbers.
- 3. Verify the Pattern: Finally, we'll verify that the pattern holds for all given numbers in the sequence. Let me start by loading the image and closely examining the numbers presented. The numbers extracted from the image are as follows:

- • First row: 16, 15, 17, 14
- • Second row: 32, 33, 31, 34 To infer the missing data, let's examine the pattern between the numbers in each row. First Row
- • The pattern does not seem to be strictly increasing or decreasing.
- • The numbers fluctuate up and down. Second Row
- • Similar to the first row, there's no clear increment or decrement.
- • The numbers also fluctuate. Given that the numbers fluctuate, we might consider that there is an alternating pattern or another non-linear relationship. Let's compare the corresponding numbers in the two rows:
- • 16 corresponds to 32
- • 15 corresponds to 33
- • 17 corresponds to 31
- • 14 corresponds to 34 This suggests that the second row might be double the first row minus a certain number. For instance:
- • Double of 16 is 32 (16 * 2 = 32)

- • Double of 15 is 30, plus 3 gives us 33 (15 * 2 + 3 = 33)

- • Double of 17 is 34, minus 3 gives us 31 (17 * 2 - 3 = 31)

- • Double of 14 is 28, plus 6 gives us 34 (14 * 2 + 6 = 34) The alternating addition and subtraction pattern could be the key here. We will attempt to double the first row's numbers and add or subtract a number to match the second row, noticing the pattern of alternation. Let me calculate the missing numbers using this hypothesis. The pattern suggests that the missing numbers are 14 for the first row and 31 for the second row. This fits with the alternating pattern of doubling the first row and then adding or subtracting a certain value to get the second row's numbers. So, the completed sequences would be:

- • First row: 16, 15, 17, 14, 14
- • Second row: 32, 33, 31, 34, 31 This means the missing number in the first row should be 14, and the corresponding missing number in the second row should be 31.

[Figure 386]

###### SPHINX

The first row has 16, 15, 17, 14. The second row has 32, 33, 31, 34.

Illustration of GPT-4V’s ability to turn at highway ramp. Green highlights the right answer in understanding. Red highlights the wrong answer in understanding. Yellow highlights the incompetence in performing the task. Check

- Figure 53: Results on Raven’s Progressive Matrices. Green highlights the right answer. Red

- highlights the wrong answer. Refer to Section 3.2 for detailed discussions.

- Section 3.2 for detailed discussions.

###### Sec. 3.3 Science Problem-solving

###### Mathematical Problems

###### Prompt

Prompt

Calculate the value of side BC step by step.

Step by step, calculate the value of y in the system of equations.

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

###### Gemini Pro

Gemini Pro

- 1. Subtract the first equation from the second equation.

2y + x = 4

- x + 1 = 3

--------

- y + 0 = 1

- 2. Solve the resulting equation for y. y = 1
- 3. Substitute the value of y back into one of the original equations to solve for x.

2y + x = 4 2(1) + x = 4 2 + x = 4 x = 2

- 4. Check your solution by substituting the values of x and y back into both original equations.

In the triangle ABC, we have:

- ∠A = 90°
- ∠B = 60° a = 3

We can use the sine function to find the length of side BC:

sin(∠B) = opposite/hypotenuse sin(60°) = 3/BC BC = 3/sin(60°) BC = 3 * 0.866 BC = 2.6

Therefore, the value of side BC is 2.6.

[Figure 391]

###### GPT-4V

[Figure 392]

x + 1 = 3

- 2 + 1 = 3
- 3 = 3

2y + x = 4 2(1) + 2 = 4 2 + 2 = 4

- 4 = 4

Therefore, the solution to the system of equations is (x, y) = (2, 1).

[Figure 393]

GPT-4V

The value of y in the given system of equations is 1.

[Figure 394]

###### SPHINX

[Figure 395]

###### SPHINX

To solve the system of equations, we need to isolate the variable y on one side of the equation. Here are the steps to solve the system: Start by adding the two equations: 2x + 1 = 3 and 2x + 4 = 3. Combine like terms: 4x + 5 = 3. Subtract 3 from both sides: 4x = -2. Divide both sides by 4: x = -2/4. Simplify the expression: x = -1/2. Therefore, the value of y in the system of equations is -1/2.

To calculate the value of side BC, we need to follow the steps outlined in the problem statement. Draw a right triangle with side lengths 60 and 3.

- Draw a line from point A to point B to form a 90-degree angle.

- Draw a line from point B to point C to form a 90-degree angle.

- Draw a line from point C to point A to form a 90-degree angle.……

- Figure 54: Results on mathematical problem-solving. Green highlights the right answer. Red highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to

- Section 3.3 for detailed discussions.

###### Sec. 3.3 Science Problem-solving

###### Mathematical Problems

Prompt

Answer the question in the figure.

[Figure 396]

[Figure 397]

###### Gemini Pro

(d) [ -2 1 0 ; -10 5 1 ]

[Figure 398]

###### GPT-4V

[Figure 399]

[Figure 400]

###### SPHINX

The matrix A is invertible if and only if the determinant of A is not equal to zero. The determinant of a matrix is a scalar value that can be calculated using the formula: |A| = ad - bc, where a, b, and c are the cofactors of the matrix A. The cofactors of a matrix are the elements of the inverse of the matrix. Therefore, to determine if a matrix is invertible, we need to find the inverse of the matrix. The inverse of a matrix is a matrix that has the same size as the original matrix and whose determinant is equal to one. The inverse of a matrix can be calculated using the formula: |A^-1| = ad - bc, where a, b, and c are the cofactors of the matrix A.

- Figure 55: Results on mathematical problem-solving. Green highlights the right answer. Red

- highlights the wrong answer. Refer to Section 3.3 for detailed discussions.

###### Sec. 3.3 Science Problem-solving

Mathematical problem

Prompt

Calculate the value of the integral equation step by step.

Prompt

Simplify this indefinite integral step by step.

[Figure 401]

[Figure 402]

[Figure 403]

Gemini Pro

[Figure 404]

[Figure 405]

Gemini Pro

[Figure 406]

[Figure 407]

GPT-4V

[Figure 408]

GPT-4V

[Figure 409]

[Figure 410]

[Figure 411]

###### SPHINX

The integral equation is written as: ∫(ln(x))dx. To calculate the value of this integral, we can use the following steps: Start by evaluating the indefinite integral of the logarithmic function. The indefinite integral of the logarithmic function is given by the formula: ln(x) + C, where C is a constant. Substitute the indefinite integral of the logarithmic function into the integral equation. This gives us: ∫(ln(x) + C)dx. Evaluate the integral using the Fundamental Theorem of Calculus. The Fundamental Theorem of Calculus states that the indefinite integral of a function can be evaluated by integrating the antiderivative of the function from 0 to infinity and adding the constant of integration. In this case, the antiderivative of the logarithmic function is ln(x) + C. Substitute the antiderivative of the logarithmic function into the integral equation. This gives us: ∫(ln(x) + C)dx = ∫(ln(x) + C) from 0 to infinity. Evaluate the integral using the limit definition of the definite integral. The definite integral of a function is the difference between the indefinite integral of the function from a to infinity and the indefinite integral of the function from 0 to a. In this case, the indefinite integral of the logarithmic function from 0 to infinity is ln(x) + C, and the indefinite integral of the logarithmic function from 0 to a is ln(a). Substitute the values of the indefinite integrals into the integral equation. This gives us: ∫(ln(x) + C)dx = ln(a) + C. Solve for C by subtracting the indefinite integral of the logarithmic function from the definite integral of the logarithmic function. This gives us: C = ln(a) - ln(x). Therefore, the value of the integral equation is ln(a) - ln(x).

[Figure 412]

SPHINX

To simplify the indefinite integral, we need to find the antiderivative of the given function. The function is given as f(x) = x^2 + 2x + 1. The antiderivative of f(x) = x^2 + 2x + 1 is given by:

F(x) = x^3/3 + x^2/2 + x/3 + C where C is the constant of integration. To simplify the expression, we can use the substitution x = sin^2(t) to express the function in terms of the sine function.

F(t) = (sin^2(t))^3/3 + (sin^2(t))^2/2 + (sin^2(t))/3 + C Now, we can integrate this expression using the Fundamental Theorem of Calculus.

F(t) = (1/3)sin^3(t) + (1/2)sin^2(t) + (1/3)sin(t) + C Therefore, the indefinite integral of f(x) = x^2 + 2x + 1 is given by:

F(x) = (1/3)sin^3(x) + (1/2)sin^2(x) + (1/3)sin(x) + C where C is the constant of integration.

- Figure 56: Results on mathematical problem-solving. Green highlights the right answer. Red highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to

- Section 3.3 for detailed discussions.

### Sec. 3.3 Science Problem-solving

###### Mathematical problem

Prompt

Answer the question in the figure. Provide intermediate steps.

[Figure 413]

[Figure 414]

###### Gemini Pro

In the right triangle on the left, the tangent of the angle is the ratio of the opposite side to the adjacent side.

[Figure 415]

Therefore, the tree is 20 feet tall.

[Figure 416]

###### GPT-4V

The tree is 20 meters tall. This is because the triangle formed by the tree and its shadow is a rightangled isosceles triangle, with the angle of elevation to the top of the tree from the tip of the shadow being 45 degrees. For such a triangle, the height is equal to the base, which is the length of the shadow. Thus, the height of the tree is the same as the length of the shadow: 20 meters.

[Figure 417]

###### SPHINX

The tree is 45 degrees tall. The tree is 45 degrees tall.

Figure 57: Results on mathematical problem-solving. Red highlights the wrong answer. Refer to Section 3.3 for detailed discussions.

#### Sec. 3.3 Science Problem-solving

###### Mathematical problem

Prompt

Prompt

Calculate the value of this expression .please think step by step.

Calculate the value of this expression.

[Figure 418]

[Figure 419]

[Figure 420]

Gemini Pro

8 + 2 - 3 x 3

= 8 + 2 - 9

[Figure 421]

= 10 - 9

###### Gemini Pro

= 1

The answer is 1.

[Figure 422]

###### GPT-4V

[Figure 423]

GPT-4V

[Figure 424]

The expression in the image is calculated as follows:

- 1. First, perform the multiplication: 3 × 3 = 9
- 2. Then, perform the addition and subtraction from left to right: 8 + 7 - 9 This simplifies to 15 - 9 which equals 6. Therefore, the value of the expression is 6.

[Figure 425]

###### SPHINX

[Figure 426]

###### SPHINX

The expression is 8 + 2 - 3 x 3 = 11.

The expression is 8 + 2 - 3 x 3 = ?. The first step is to solve the expression inside the parentheses. 8 + 2 - 3 x 3 = 8 + 2 - 9 = 1. The expression is 8 + 2 - 3 x 3 = 1.

Figure 58: Results on mathematical problem-solving. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 3.3 for detailed discussions.

##### Sec. 3.3 Science Problem-solving

###### Mathematical problem

###### Prompt

Prompt

Is the answer to the arithmetic question in the image 32? Please answer yes or no.

What is the answer to the arithmetic question in the image?

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

###### Gemini Pro

###### Gemini Pro

yes

The answer is 33.

[Figure 431]

[Figure 432]

###### GPT-4V

###### GPT-4V

No, the answer to the arithmetic question in the image is not 32.

The answer to the arithmetic question in the image is 33.

[Figure 433]

[Figure 434]

###### SPHINX

SPHINX

No

The answer to the arithmetic question is 10.

Figure 59: Results on mathematical problem-solving. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 3.3 for detailed discussions.

###### Sec. 3.3 Science Problem-solving

###### Physics Problems

Prompt

Solve this physics problem step by step.

[Figure 435]

[Figure 436]

###### Gemini Pro

The average speed of the student during the entire race can be calculated using the formula: average speed = total distance / total time In this case, the total distance is 100 meters, and the total time is 10 seconds. Therefore, the average speed is: average speed = 100 meters / 10 seconds = 10 meters per second So the answer is 10 m/s

[Figure 437]

###### GPT-4V

The average speed of the student during the entire 100-meter dash is 10 meters per second.

[Figure 438]

###### SPHINX

The problem states that a runner in a 100-meter dash, starts at the start of the race, and 10 seconds later, when crossing the finish line, the instantaneous speed is 8 meters per second. To solve this problem, we need to calculate the average speed of the runner during the entire race. The average speed of the runner during the entire race is calculated by dividing the total distance traveled by the runner (100 meters) by the total time taken by the runner to complete the race (10 seconds). Therefore, the average speed of the runner during the entire race is 100 meters / 10 seconds = 10 meters per second.

- Figure 60: Results on physics problem-solving. Green highlights the right answer. Refer to Section 3.3 for detailed discussions.

###### Sec. 3.3 Science Problem-solving

Physics Problems

###### Prompt

Prompt

Solve this physics problem step by step.

[Figure 439]

Here is a solution to a Physics problem by a student. Try to follow the steps already taken to complete the solution. Make sure to use LaTeX for math. Since the problem does not provide specific values for each physical quantity, the result should only include the mathematical expression for F.

[Figure 440]

[Figure 441]

Gemini Pro

First find the total resistance of the circuit. Since the rheostat is a variable resistor, its resistance can vary. However, since the slider is positioned at its midpoint, the resistance of the rheostat is equal to half of its total resistance, which is 10 Ω. The total resistance of the circuit is then:

###### R_total = R_0 + R_rheostat

= 40 Ω + 10 Ω = 50 Ω Now use Ohm's law to find the current flowing through the circuit:

I = U_AB / R_total = 100 V / 50 Ω = 2 A Since the resistors R_0 and R_rheostat are connected in parallel, the current flowing through R_0 is:

[Figure 442]

Gemini Pro

[Figure 443]

I_R0 = I / 2 = 2 A / 2 = 1 A Therefore, the current flowing through the resistor R_0 is 1 A.

[Figure 444]

GPT-4V

[Figure 445]

[Figure 446]

GPT-4V

[Figure 447]

[Figure 448]

###### SPHINX

The problem states that the current is flowing through the resistor. The current is flowing through the resistor, so the current is flowing through the resistor. The answer is yes.

[Figure 449]

SPHINX

（the result is none.）

Note：we show the rendered result of some of the output.

- Figure 61: Results on physics problem-solving. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 3.3 for detailed discussions.

###### Sec. 3.3 Science Problem-solving

###### Physics Problems

Prompt

[Figure 450]

GPT-4V

Solve this physics problem step by step.

[Figure 451]

[Figure 452]

[Figure 453]

###### Gemini Pro

- 1. To solve this problem, we must first find the velocity of the ball at point B.
- 2. To do this, we can use the conservation of energy principle. The initial energy of the ball is all potential energy, due to its height above the ground. The final energy of the ball is all kinetic energy, due to its motion.

- 3. We can write the conservation of energy equation as follows: mgh = 1/2mv^2

where m is the mass of the ball, g is the acceleration due to gravity, h is the height of the ball above the ground, and v is the velocity of the ball.

- 4. We know that the height of the ball above the ground is R, the radius of the track. We also know that the acceleration due to gravity is 10 m/s^2. We can substitute these values into the conservation of energy equation to get:

mgR = 1/2mv^2

- 5. We can solve this equation for v to get: v = sqrt(2gR)
- 6. Now that we know the velocity of the ball at point B, we can find the distance it travels before landing.

- 7. To do this, we can use the equation of motion for a projectile: y = -1/2gt^2 + vt + h

where y is the height of the ball, g is the acceleration due to gravity, t is the time of flight, v is the velocity of the ball, and h is the initial height of the ball.

- 8. We know that the height of the ball at point B is R, the radius of the track. We also know that the acceleration due to gravity is 10 m/s^2. We can substitute these values into the equation of motion to get:

R = -1/2gt^2 + vt

- 9. We can solve this equation for t to get: t = (v + sqrt(v^2 + 2gR))/g

- 10. Now that we know the time of flight of the ball, we can find the distance it travels before landing.
- 11. To do this, we can use the equation: d = vt

where d is the distance traveled, v is the velocity of the ball, and t is the time of flight.

- 12. We know that the velocity of the ball at point B is sqrt(2gR). We also know the time of flight of the ball. We can substitute these values into the equation to get:

d = sqrt(2gR) * (v + sqrt(v^2 + 2gR))/g

- 13. We can simplify this equation to get: d = 2R + sqrt(2RgR)
- 14. We know that the radius of the track is 1 m. We can substitute this value into the equation to get:

d = 2(1) + sqrt(2(1)(10))

- 15. We can calculate this to get: d = 2 + sqrt(20)
- 16. We can calculate this to get: d = 2 + 4.47

- 17. Therefore, the landing point C is 6.47 m from point A.

[Figure 454]

[Figure 455]

###### SPHINX

Step-by-step solution: First, we need to find the distance the ball travels along the track. This distance is given by the formula:

d = vt where v is the velocity of the ball, t is the time it takes for the ball to travel from point A to point B, and d is the distance traveled. We know that the velocity of the ball is given by the formula:

v = 10 m/s^2 and the time it takes for the ball to travel from point A to point B is given by the formula:

t = 2R/v Substituting these values into the formula for the distance traveled, we get:

d = 2R/10 m/s^2 = 0.05 m This is the distance the ball travels along the track. Next, we need to find the distance from point A to point C. This distance is given by the formula:

###### C = A + d

Substituting the values for A, d, and R, we get:

C = A + 0.05 m This is the distance from point A to point C. Finally, we need to find the angle between point A and point C. This angle is given by the formula:

[Figure 456]

GPT-4V

###### θ = 180°/π (C/A)

[Figure 457]

Substituting the values for A, C, and R, we get:

θ = 180°/π (0.05 m/1 m) = 180°/π This is the angle between point A and point C. Therefore, the distance from point A to point C is 0.05 m, and the angle between point A and point C is 180°/π radians.

Note：The output of GPT-4V is the Latex code as well as the output of Gemini is the Markdown code and we show the rendered result.

- Figure 62: Results on physics problem-solving. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 3.3 for detailed discussions.

###### Sec. 3.4 Emotion Understanding

###### Facial Expression Analysis

[Figure 458]

Gemini Pro

###### Prompt

The expression on the face is a happy one. The leaf is smiling.

Please analyze this expression and what emotions it wants to express.

[Figure 459]

[Figure 460]

GPT-4V

The image you've provided appears to show a concrete surface with two dark holes and a red leaf positioned in such a way that it resembles a face: the holes act as the eyes and the leaf as a mouth. This kind of arrangement is often interpreted as a form of pareidolia, where our minds perceive a familiar pattern of a face where there isn't one.

The 'expression' of the face seems neutral or nonemotive; however, interpretations of this kind of imagery can vary greatly from person to person. Some might see it as a 'smiling' face due to the curvature of the leaf, suggesting a sense of whimsy or happiness. Others might interpret it as a 'frowning' face, depending on the angle and their perspective, which could convey a sense of sadness or melancholy.

[Figure 461]

SPHINX

The overall emotion that this image wants to express may be open to interpretation, and it could be seen as a reflection of the viewer's current mood or feelings. It doesn't express emotion in the human sense but invites the viewer to project emotion onto it. This kind of visual play is common in photography and art, providing an engaging visual puzzle that encourages the viewer to stop and consider the image more deeply.

The image features a red leaf placed on a cement surface, which has two holes in it. The leaf appears to be smiling, giving the impression of a happy and playful expression. This creative arrangement of the leaf and the holes in the cement can evoke feelings of joy, whimsy, and a sense of lightheartedness. The artist may have intended to bring a smile to the viewer's face or to encourage them to see the beauty in everyday objects and situations.

###### Prompt

[Figure 462]

Gemini Pro

Identify and read emotions of people from their faces as shown in the image.

The person in the image is scared. The person has his hands on his face, with his mouth open. His eyes are wide open and his eyebrows are raised. These are all signs of fear.

[Figure 463]

[Figure 464]

GPT-4V

The person in the image appears to display an expression of fear or intense anxiety. Their eyes are wide open, eyebrows are raised, and mouth is slightly open, which are all facial cues commonly associated with fear. The person is also biting their nails, a gesture often linked to nervousness or stress. These visual cues can indicate that the person is experiencing a moment of significant distress or apprehension.

[Figure 465]

SPHINX

I'm sorry, but I cannot see any faces or emotions in the image. The image only contains text.

- Figure 63: Results on facial expression analysis. Green highlights the right answer. Red highlights

the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 3.4 for detailed discussions.

###### Sec. 3.4 Emotion Understanding

###### Facial Expression Analysis

###### Prompt

[Figure 466]

###### GPT-4V

Please analyze this expression and what emotions it wants to express.

The image you've uploaded appears to be a coffee cup with a frowning face on the surface of the coffee, with the rim of the cup reading "HAVE A GOOD DAY". The expression on the coffee seems to be one of sadness or dissatisfaction, which contrasts with the message on the cup.

[Figure 467]

The frowning face is a universal symbol of unhappiness or discontent. When combined with the text "HAVE A GOOD DAY", it might suggest an ironic or sarcastic message, playing on the discrepancy between the unhappy expression and the typically positive sentiment of the phrase. It could be interpreted as a humorous take on a bad day, or expressing the idea that even if things aren't great, one can still find a reason to smile or hope for a better day.

[Figure 468]

[Figure 469]

###### Gemini Pro SPHINX

The expression "Have a frown day" is a playful and humorous way of saying "Have a bad day." The frown face in the coffee foam is meant to represent a negative emotion or experience, and the phrase is intended to be lighthearted and not taken seriously. It is a creative way of expressing a difficult or unfortunate day, and the intention is to evoke a sense of humor and resilience in the face of adversity.

It looks like the coffee is expressing a feeling of tiredness or boredom. The eyes are half-closed and the mouth is turned down at the corners. The overall impression is one of sadness or resignation.

- Figure 64: Results on facial expression analysis. Green highlights the right answer. Yellow highlights the incompetence in performing the task. Refer to Section 3.4 for detailed discussions.

###### Sec. 3.4 Emotion Understanding

###### Image Emotion Analysis

Prompt

What feelings and emotions do people have about this picture?

[Figure 470]

[Figure 471]

###### Gemini Pro

This picture makes me feel sad and lonely. The dark clouds and the empty road make it look like there is no hope. The only light comes from the street lamps, which are far away and barely visible. This picture makes me feel like I am all alone in the world.

[Figure 472]

###### GPT-4V

The image you've shared features a rather moody and overcast sky, which may evoke a variety of feelings and emotions in different people. Here are some possible reactions:

- 1. Melancholy or Calmness: The gray tones and the overcast nature of the sky can induce a feeling of melancholy or introspection in some. It may also bring a sense of calmness, as overcast days often bring a quietude to the environment.

- 2. Mystery or Foreboding: The dark clouds could be seen as foreboding, suggesting an impending storm or bad weather. This could create a sense of tension or apprehension about what is to come.

- 3. Reflection or Peacefulness: For others, such scenes can be reflective, providing a moment to pause and think. The absence of bright sunshine and the presence of diffused light can be soothing and offer a break from the hustle of sunny days.

- 4. Appreciation of Nature: Some might find beauty in the drama of the sky and the silhouettes of the cranes and trees, appreciating the natural light and the landscape's mood.

- 5. Disappointment or Annoyance: If someone were hoping for a sunny day, this scene might be a disappointment. It could also be an annoyance for those with outdoor plans, as overcast conditions might imply a change of plans.

It's important to note that reactions to images are highly subjective and can be influenced by a person's mood, experiences, and personal preferences.

[Figure 473]

###### SPHINX

People might experience a sense of calm, tranquility, or even melancholy when looking at this picture. The image features a dark, cloudy sky over a city, with a bridge and construction cranes in the background. The combination of the dark sky and the silhouette of the cityscape can evoke feelings of introspection or contemplation. The presence of the construction cranes might also suggest a sense of progress or development, which could be seen as a positive aspect. However, the overall mood of the picture is primarily influenced by the dark, cloudy sky, which can evoke a sense of calmness or even sadness in some people.

Illustration of GPT-4V’s ability to turn at highway ramp. Green highlights the right answer in understanding. Red highlights the wrong answer in understanding. Yellow highlights the incompetence in performing the task. Check Section 3.4 for detailed discussions.

- Figure 65: Results on image emotion analysis. Green highlights the right answer. Refer to Section 3.4 for detailed discussions.

###### Sec. 3.4 Emotion Understanding

###### Image Emotion Analysis

###### Prompt

Prompt

What feelings and emotions do people have about this picture?

What feelings and emotions do people have about this picture?

[Figure 474]

[Figure 475]

[Figure 476]

Gemini Pro

[Figure 477]

###### Gemini Pro

This picture makes me feel calm and peaceful. The blue water is so clear and inviting. I feel like I could just float around in this water all day. The fish are all swimming around so peacefully. It's just a beautiful and serene scene.

The picture is of a beautiful green field with cows grazing. The sky is blue and there are some clouds. The picture evokes feelings of peace, tranquility, and happiness. The cows are a symbol of abundance and fertility. The green field is a symbol of growth and new beginnings. The blue sky is a symbol of hope and optimism. The clouds are a symbol of change and transformation. The picture is a reminder that even in the midst of change, there is always beauty to be found.

[Figure 478]

###### GPT-4V

This underwater image can evoke a variety of feelings and emotions:

- • Awe and Wonder: The vast blue underwater scene with the diver and schools of fish can instill a sense of wonder about the ocean's immensity and biodiversity.

- • Tranquility: The deep blue of the water and the gentle gliding of the fish can be very calming and peaceful.

- • Adventure and Exploration: For those who love the ocean or diving, this scene might evoke the thrill of exploring unknown parts of the world.

- • Isolation: The diver appears alone in the vast ocean which might evoke feelings of solitude.

- • Appreciation for Nature: Observing marine life in its natural habitat can lead to an appreciation of the beauty of nature and the need to protect it.

- • Insignificance: The enormity of the ocean and the small size of the diver can make one reflect on the vastness of the natural world compared to the individual.

[Figure 479]

###### GPT-4V

The image depicts a pastoral scene with cows grazing on a vibrant green landscape under a blue sky scattered with white clouds. Such a scene often evokes feelings of tranquility, peace, and a sense of escape from the hustle and bustle of urban life. It may also inspire a sense of nostalgia for the countryside or for a simpler, more natural way of life. For some, it can evoke feelings of joy or contentment, being a reminder of the beauty and serenity found in nature. Others might feel a sense of longing or a desire to be outdoors and connected with the environment. The image can also be seen as a representation of sustainability and harmony between humans and nature.

[Figure 480]

The emotions one might feel can be deeply personal and vary widely from person to person.

SPHINX

People might feel a sense of tranquility, peace, and connection to nature when looking at this picture. The image depicts a herd of cows grazing in a lush green field, surrounded by rolling hills and a beautiful blue sky. The serene landscape and the presence of the cows, which are often associated with rural life and agriculture, can evoke feelings of calmness and relaxation. Additionally, the image might also remind people of the importance of sustainable farming practices and the role of animals in our ecosystem.

[Figure 481]

SPHINX

People might experience feelings of awe, wonder, and excitement when looking at this picture. The image showcases a person scuba diving in the ocean, surrounded by a large school of fish. The vastness of the underwater world, the abundance of fish, and the thrill of exploring such an environment can evoke feelings of adventure and curiosity. Additionally, the presence of the scuba diver adds a sense of connection to nature and the underwater world, which can inspire feelings of appreciation and respect for the natural world.

Illustration of GPT-4V’s ability to turn at highway ramp. Green highlights the right answer in understanding. Red highlights the wrong answer in understanding. Yellow highlights the incompetence in performing the task. Check Section 3.4 for detailed discussions.

- Figure 66: Results on image emotion analysis. Green highlights the right answer. Refer to Section 3.4 for detailed discussions.

###### Sec. 3.4 Emotion Understanding

###### Image Emotion Analysis

Prompt

What feelings and emotions do people have about this picture?

[Figure 482]

[Figure 483]

###### Gemini Pro

The picture is of a couple sitting on a bench, looking out at the Northern Lights. The sky is dark, and the water is still. The couple is holding each other close, and they look happy and content. The picture evokes a feeling of peace and tranquility. The couple is surrounded by natural beauty, and they are enjoying each other's company. The picture is a reminder that there are still places in the world where people can escape the hustle and bustle of everyday life and connect with nature. The picture also evokes a feeling of romance. The couple is clearly in love, and they are enjoying each other's company. The picture is a reminder that love is still alive and well in the world. Overall, the picture is a beautiful and evocative reminder of the power of nature and the importance of love.

[Figure 484]

###### GPT-4V

The image you've shared appears to be a candid moment of two individuals sitting on a bench, looking out at a beautiful display of the Northern Lights. The scene evokes a sense of tranquility and companionship. Observers might feel a range of emotions such as:

- 1. Wonder and Awe: The natural spectacle of the aurora borealis can inspire profound feelings of amazement and admiration for the beauty of nature.

- 2. Intimacy and Connection: The close proximity of the two individuals suggests a personal bond, which might resonate with viewers as a tender display of human connection.

- 3. Peacefulness: The quiet and serene setting, away from the bustle of city life, may evoke a sense of calm and peacefulness.

- 4. Reflection: The vastness of the night sky, coupled with the rare natural phenomenon, might prompt introspection or philosophical reflection about one's place in the universe.

- 5. Longing: For some, this image could evoke a longing or desire to experience such a moment themselves, whether it be the natural event or the shared experience with someone else.

The specific emotions an individual feels may vary based on personal experiences, desires, and the context in which they view the image.

[Figure 485]

###### SPHINX

People might experience a range of feelings and emotions when looking at this picture. The image shows a couple sitting on a log, watching the northern lights in the sky. The northern lights are often associated with awe, wonder, and a sense of connection with nature. The couple's shared experience of witnessing this natural phenomenon might evoke feelings of closeness, intimacy, and shared memories. The serene and peaceful setting could also inspire feelings of relaxation, tranquility, and appreciation for the beauty of the world.

Illustration of GPT-4V’s ability to turn at highway ramp. Green highlights the right answer in understanding. Red highlights the wrong answer in understanding. Yellow highlights the incompetence in performing the task. Check Section 3.4 for detailed discussions.

- Figure 67: Results on image emotion analysis. Green highlights the right answer. Refer to Section 3.4 for detailed discussions.

###### Sec. 3.4 Emotion Understanding

###### Image Emotion Analysis

Prompt

Aesthetics represents human aesthetics of all things, including design, emotion, and visual attraction. Considering societal standards and norms, which of the two images would be more likely to be regarded as ha ving higher aesthetic value by a majority of people? Please explain your choice.

[Figure 486]

[Figure 487]

[Figure 488]

###### Gemini Pro

The first image is more likely to be regarded as having higher aesthetic value by a majority of people because it conforms to traditional societal standards and norms of beauty. The bridge in the first image has a more natural, rustic feel, with its stone construction and simple design. The setting is also more natural, with trees and a river in the background. The second image, on the other hand, is of a modern, concrete bridge, which may be seen as less aesthetically pleasing by some people because it is more industrial and man-made. Additionally, the setting of the second image is more urban, with buildings and other structures in the background, which may be seen as less appealing than the natural setting of the first image.

[Figure 489]

###### GPT-4V

Determining aesthetic value can be subjective and varies greatly among individuals based on their cultural background, personal preferences, and the context in which they encounter the work. Both images feature bridges, which can be appreciated for their architectural design, their integration into the surrounding environment, and the feelings they may evoke.

The first image presents an arch bridge, often associated with historical and classical aesthetics. The use of stone, the gentle curve of the arch, the reflection on the water, and the surrounding greenery might be perceived as harmonious and tranquil. This type of bridge can be valued for its traditional craftsmanship and natural integration into the landscape.

The second image features a suspension bridge, a symbol of modern engineering and design. Its towering pillars and the complex network of cables against the backdrop of the mountains and the clear sky might be seen as a testament to human innovation and might appeal to those who value modernism and industrial strength.

Ultimately, the image regarded as having a higher aesthetic value would depend on the viewer. Some may prefer the classical beauty and natural setting of the arch bridge, while others may find the grand scale and modern design of the suspension bridge more aesthetically pleasing. Additionally, societal standards and norms around beauty and aesthetics can shift over time and differ across cultures, making a definitive answer challenging. However, in a broad contemporary context, the suspension bridge might be considered to have a higher aesthetic value due to its scale, the marvel of its construction, and its representation of modern achievements.

[Figure 490]

###### SPHINX

The first image has a wider bridge and more people in it, while the second image has a more scenic view with a river and mountains in the background.

The results of different models for Image Emotion Analysis. Green highlights the right answer in understanding. Yellow highlights the incompetence in performing the task.

- Figure 68: Results on image emotion analysis. Green highlights the right answer. Yellow highlights the incompetence in performing the task. Refer to Section 3.4 for detailed discussions.

###### Sec. 3.4 Emotion Understanding

###### Image Emotion Analysis

Prompt

Aesthetics represents human aesthetics of all things, including design, emotion, and visual attraction.Consider ing societal standards and norms, which of the two images would be more likely to be regarded as having high er aesthetic value by a majority of people? Please explain your choice.

[Figure 491]

[Figure 492]

[Figure 493]

###### Gemini Pro

The second image would be more likely to be regarded as having higher aesthetic value by a majority of people. This is because it is a more visually appealing image. The first image is of a rural area, and the second image is of a cityscape. The cityscape is more likely to be seen as more aesthetically pleasing because it is more vibrant and has more interesting features. The first image is more muted and has fewer features that would be considered visually appealing. Additionally, the second image is of a city at night, which is often seen as more romantic and beautiful than a city during the day.

[Figure 494]

###### GPT-4V

The question of which image would be considered to have higher aesthetic value by a majority of people is quite subjective, as aesthetic preferences can vary widely based on individual tastes, cultural backgrounds, and contexts. However, societal standards and norms often relate aesthetic value to certain characteristics like harmony, balance, vividness, and the elicited emotional response.

The first image depicts a landscape view from a high vantage point on a cloudy day, providing a sense of tranquility and the natural environment. The colors are subdued, and the atmosphere is calm. It can be associated with the aesthetic principles of simplicity and natural beauty.

The second image shows a nighttime cityscape, full of vibrant lights and a variety of architectural designs. This image might be associated with human achievement, modernity, and the energy of urban life. The bright colors and the dynamic light contrast can be visually stimulating.

In a contemporary context, the second image might be more likely to be regarded as having higher aesthetic value by a majority due to its vibrant colors, the complexity and grandeur of the urban environment, and the fact that night images with artificial lighting can often appear more dramatic and engaging. The appeal of city lights and modern architecture is commonly celebrated in media and may resonate with societal fascinations with technology and progress.

However, this does not undermine the potential for the first image to be highly valued for its serene and unspoiled natural landscape, which might be preferred by those who value nature and peacefulness in aesthetics.

Ultimately, aesthetic value is personal and culturally influenced, so while one can speculate based on certain criteria, the actual preferences could vary greatly among different people.

[Figure 495]

###### SPHINX

The second image, which shows a city skyline at night with bright lights, would be more likely to be regarded as having higher aesthetic value by a majority of people. This is because city skylines are often associated with modern architecture, vibrant nightlife, and bustling urban environments, which are generally perceived as more visually appealing and emotionally engaging than rural landscapes. Additionally, the bright lights and illuminated buildings in the city skyline create a sense of excitement and energy that is more attractive to many people.

The results of different models for Image Emotion Analysis. Green highlights the right answer in understanding. Yellow highlights the incompetence in performing the task.

- Figure 69: Results on image emotion analysis. Green highlights the right answer. Refer to Section 3.4 for detailed discussions.

###### Sec. 3.4 Emotion Understanding

###### Emotion Conditioned Output

[Figure 496]

###### Prompt

Gemini Pro

Describe the image in a romantic way.

The golden sun sets behind the tall buildings, casting a warm and romantic glow over the city street. A cyclist rides by, enjoying the last few moments of daylight. A young woman walking her bike follows, her hair blowing in the breeze. The scene is peaceful and serene, a perfect moment captured in time.

[Figure 497]

[Figure 498]

GPT-4V

The sun dips low in the sky, casting a warm, golden hue over the cityscape. It's the golden hour, where everything takes on a hue of honey and amber. Shadows stretch long across the pavement, and the everyday hustle of the city seems to slow just a bit as people make their way home. The sun, a fiery globe, hangs just above the horizon, caught between the silhouettes of buildings and trees, as if pausing to bid farewell before it retreats for the night. It's a moment of tranquility and beauty, a reminder of nature's quiet spectacle amidst the urban environment.

[Figure 499]

SPHINX

The sun sets over the city, casting a warm glow over the street. People are walking down the sidewalk, some carrying backpacks, while others ride bicycles. The street is lined with parked cars, and the buildings in the background create a picturesque urban landscape. The scene is peaceful and serene, as the sun dips below the horizon, painting the sky with a beautiful array of colors.

Figure 70: Results on emotion-conditioned output. Yellow highlights the incompetence in performing the task. Refer to Section 3.4 for detailed discussions.

###### Sec. 3.4 Emotion Understanding

###### Emotion Conditioned Output

###### Prompt

[Figure 500]

Gemini Pro

Describe the image in a terrifying way.

The city is dark and gloomy. The only light comes from the streetlights and the buildings. The clouds

[Figure 501]

are thick and heavy, and they seem to be pressing down on the city. There is no moon, and the stars are hidden behind the clouds. The only sound is the faint hum of traffic. It is a perfect night for a crime.

[Figure 502]

###### GPT-4V

The city stretches out beneath a sinister sky, the darkness almost tangible as it presses down on the myriad of buildings below. The lights that usually signify life and activity now flicker uncertainly, casting an eerie, otherworldly glow on the streets that twist and turn like a labyrinth meant to trap the unwary. The overcast heavens loom ominously, as if hiding a malevolent force just beyond human sight, waiting for the right moment to descend. Shadows seem to creep and move of their own accord, and the air is thick with a foreboding silence, broken only by the occasional distant sound that could be the wind—or perhaps something far less benign. It's a scene that whispers of secrets and dangers, of a city on the edge of an abyss, with the night holding its breath for the unspeakable to unfold.

[Figure 503]

SPHINX

The image depicts a dark and stormy night over a city skyline. The sky is filled with dark clouds, and the city lights are flickering, creating an eerie atmosphere. The buildings in the city are illuminated, but the overall scene is shrouded in darkness. The night sky is punctuated by the occasional lightning strike, casting a eerie glow on the cityscape. The combination of the dark clouds, flickering lights, and the ominous presence of lightning creates a terrifying and unsettling scene.

- Figure 71: Results on emotion-conditioned output. Yellow highlights the incompetence in performing the task. Refer to Section 3.4 for detailed discussions.

###### Sec. 3.5 Playing Game

###### Sudoku

###### Prompt

Prompt

The data in the grid is provided below:

This is a Sudoku game. You need to fill in the numbers 1-9 in the 9×9 grid so that every column, row, and every 3×3 subgrid contains all the numbers 1-9. Please fill in the missing numbers in the grid.

- 2 1 9 4 3 5 8 7 6

- 8 5 2 6 1 3 4 9

3 6 4 7 8 9 1 5 2 6 4 3 5 9 7 2 8 1

- 9 7 1 8 2 3 4 6 5 2 8 1 4 6 7 9 3

- 4 5 2 6 1 8 9 3 7 1 9 7 3 5 4 6 2 8 …(The text and image are the same as the previous prompt)

[Figure 504]

8 3 6 9 7 2 5 1 4

[Figure 505]

Gemini Pro

…(The text and image are the same as the previous prompt) The data in the grid is provided below:

[Figure 506]

[Figure 507]

###### SPHINX

###### Gemini Pro

- 2 1 9 4 3 5 8 7 6

- 7 8 5 2 6 1 3 4 9

3 6 4 7 8 9 1 5 2

- 6 4 3 5 9 7 2 8 1 9 7 1 8 2 3 4 6 5

- 5 2 8 1 4 6 7 9 3

8 3 6 9 7 2 5 1 4 4 5 2 6 1 8 9 3 7

- 1 9 7 3 5 4 6 2 8

It appears that the OCR process did not successfully extract the text from the image of the Sudoku puzzle. Given this, I will proceed by manually analyzing the image and solving the puzzle. Let's fill in the missing numbers in the grid. ​​The Sudoku puzzle has been solved. Here are the missing numbers filled in:

- 2 1 9 | 4 3 5 | 8 7 6

7 8 5 | 2 6 1 | 3 4 9

- 3 6 4 | 7 8 9 | 1 5 2

------+-------+-----6 4 3 | 5 9 7 | 2 8 1 9 7 1 | 8 2 3 | 4 6 5 5 2 8 | 1 4 6 | 7 9 3 ------+-------+-----8 3 6 | 9 7 2 | 5 1 4

- 4 5 2 | 6 1 8 | 9 3 7 1 9 7 | 3 5 4 | 6 2 8 row, column, and 3x3 subgrid now contains all the numbers from 1 to 9.

I'm sorry, but I cannot see the image you are referring to. Can you please provide the numbers in the 9×9 grid or describe the Sudoku game in more detail?

+-------+-------+-------+

- | 2 1 9 | 4 3 5 | 8 7 6 |

- | 8 5 2 | 6 1 3 | 4 9 7 |

| 3 6 4 | 7 8 9 | 1 5 2 |

+-------+-------+-------+ | 6 4 3 | 5 9 7 | 2 8 1 |

- | 9 7 1 | 8 2 3 | 4 6 5 | | 2 8 5 | 4 6 1 | 7 3 9 |

+-------+-------+-------+ | 8 3 6 | 9 7 2 | 5 1 4 | | 5 2 7 | 3 5 4 | 6 9 3 | | 1 9 7 | 3 5 4 | 6 2 8 | +-------+-------+-------+

[Figure 508]

###### GPT-4V

[Figure 509]

GPT-4V

It seems that the OCR was unable to correctly interpret the Sudoku puzzle from the image. Since the OCR result is empty, I will proceed to solve the Sudoku puzzle by implementing a solver algorithm that will analyze the image directly. Let's start by detecting the digits and then fill in the missing numbers. ​​The current approach to detect and recognize the digits from the Sudoku grid using contours and OCR has not been successful. All the cells are being identified as empty. This might be due to the thresholding not being well suited for the image or the cell size assumptions being incorrect. I will try a different approach to identify the grid lines and use them to isolate each cell for digit recognition. If the grid lines are uniform and cells are perfect squares, this approach should provide better results. Let's proceed with this new method. ​​Error analyzing It appears there was an error with the image dilation process due to the kernel type not being correct. The kernel used for dilation should be of type uint8 rather than float. I will correct the kernel type and try the process again to find the Sudoku grid and recognize the digits. Let's attempt this once more. ​Stopped analyzing.

[Figure 510]

SPHINX

I'm sorry, but I cannot see the image you are referring to. Can you please provide the numbers in the 9×9 grid or describe the Sudoku game in more detail?

- Figure 72: Results on Sudoku playing. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 3.5 for detailed discussions.

Illustration of GPT-4V’s ability to turn at highway ramp. Green highlights the right answer in understanding. Red highlights the wrong answer in understanding. Yellow highlights the incompetence in performing the task. Check Section 4.3 for detailed discussions.

###### Sec. 3.5 Playing Game

###### Crossword

Prompt

A crossword puzzle is a word game that involves a grid of squares. Players enter words that intersect each other horizontally ("across") and vertically ("down") based on a set of clues. Please complete the crossword puzzle.

[Figure 511]

###### GPT-4V

Let's solve the crossword puzzle based on the clues provided: Across:

[Figure 512]

- 1. A person who has received the highest university degree

- 3. All _____ lead to Rome Down:

2. A computer keyboard key

- 4. A color The highest university degree is typically a "Doctorate," which is abbreviated as "PhD." The phrase "All roads lead to Rome" refers to the fact that Rome was the center of the Roman Empire, and many of its roads spread out to the empire's far-flung regions. For 2 down, a common computer keyboard key could be "Enter" or "Shift," but we need to consider the intersection with "1 across" which is "PhD". "Enter" seems unlikely since there's no common color that starts with "R," while "Shift" could work if we consider "White" as the color for 4 down, which fits the number of boxes. Therefore, the answers could be:

[Figure 513]

Gemini Pro

ACROSS

- 1. A person who has received the highest university degree – DOCTOR

- 3. All ___ lead to Rome - ROADS DOWN

2. A computer keyboard key – DELETE

- 4. A color - RED

- 1. PhD

- 2. Shift

- 3. Roads

- 4. White Let's place these into the crossword:

[Figure 514]

SPHINX

- • 1 Across (PhD) will have 'P' in the second square, 'H' in the third, and 'D' in the fifth.
- • 2 Down (Shift) will start with 'S' in the second square of 1 Across and 'H' intersecting with 'H' in 1 Across.
- • 3 Across (Roads) will start with 'R' in the fourth square of 1 Across.
- • 4 Down (White) will start with 'W' in the fourth square of 1 Across, intersecting with 'O' from 3 Across.

- 1. A person who has received the highest university degree

- 2. A computer keyboard key

- 3. All _____ lead to Rome

- 4. A color The completed crossword puzzle should look like this:

- 1. A person who has received the highest university degree

- 2. A computer keyboard key

- 3. All _____ lead to Rome

- 4. A color The completed crossword puzzle should look like this:

I will fill out the crossword accordingly.

- Figure 73: Results on Crossword playing. Green highlights the right answer. Red highlights the

Illustration of GPT-4V’s ability to turn at highway ramp. Green highlights the right answer in understanding. Red highlights the wrong answer in understanding. Yellow highlights the incompetence in performing the task. Check Section 3.6 for detailed discussions.

wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 3.5 for detailed discussions.

###### Sec. 3.5 Playing Game

###### Crossword

###### Prompt

[Figure 515]

###### GPT-4V

A crossword puzzle is a word game that involves a grid of squares. Players enter words that intersect each other horizontally ("across") and vertically ("down") based on a set of clues. Please complete the crossword puzzle.

To solve the crossword puzzle you've provided, we need to fill in the answers to the clues given. Here are the clues and their likely answers: Across:“ An ___ a day, keep doctor away"

[Figure 516]

• The common saying is "An apple a day keeps

the doctor away."“ Very large"

• This could have many answers like "huge," "giant," or "massive," but without knowing the number of letters, it's hard to be certain.

Down:

2. "An animal"

• This is very broad, and without knowing the number of letters or having intersecting letters from other answers, it's impossible to determine the exact answer.

"True or ___"

• The common expression is “True or false.” Let's place the answers we know into the puzzle: Across, starting from the first square on the top left, the word "apple" fits perfectly. Down, starting from the fourth square on the top row, the word "false" fits. With the word "apple" in place, we now know that the second letter for 2. Down is "p," and the third letter is "p" as well. However, without the number of squares to fill, we cannot determine a specific animal. Similarly, with "false" in place for 4. Down, we know the second letter for 3. Across is "a." If we assume the word for 3. Across is five letters to match the squares, "large" could be a fitting answer, but this is just a guess. Would you like to proceed with these answers? If the number of squares for each answer is given, that would help in determining the most accurate responses.

[Figure 517]

Gemini Pro

Across:

- 1. An apple a day, keep doctor away

- 3. Very large Down:

2. An animal

- 4. True or false

[Figure 518]

SPHINX

An _____ a day, keep doctor away An animal Very large True or _____ Answer: An apple A cat Very large True or false

- Figure 74: Results on Crossword playing. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 3.5 for detailed discussions.

###### Sec. 3.5 Playing Game

###### Go Playing

Prompt

Assuming you are a Go player, given a particular position as shown in the diagram, you need to play a black piece next, think step by step about which position you should play. The result is described by coordinates, e.g. A1.You can choose a position with the highest level of confidence.

[Figure 519]

[Figure 520]

###### Gemini Pro

The result is described by coordinates, e.g. A1.You can choose a position with the highest level of confidence. The best move for black is at C4.

[Figure 521]

###### GPT-4V

To analyze this Go position, I'll need to take a step-by-step approach, considering strategic aspects such as influence, territory, and life and death of groups. This will involve looking at the stability of the black and white groups, potential for expansion, and any immediate threats that need to be addressed. However, I must mention that my capabilities as an AI are based on patterns and knowledge, and while I can give an informed suggestion, I do not have the ability to actively play the game or simulate moves like specialized Go software would. With the current position, the most important aspects to consider for Black would be:

- 1. Strength of Groups: Ensuring that the black groups are stable and have two eyes.
- 2. Cutting Points: Identifying any cutting points that White could exploit to separate Black's stones.
- 3. Influence vs. Territory: Deciding whether to play to increase influence (control of the center) or to secure territory (edges and corners).
- 4. Sente and Gote: Determining whether the move is sente (forcing the opponent to respond) or gote (ending the interaction, allowing the opponent to take the initiative). Taking into account these aspects, I would analyze the board and suggest a move. However, since I'm unable to actively simulate the game, I can't provide a definitive best move without the ability to calculate and anticipate subsequent moves from the opponent. In actual play, it's best to use a Go software or consult with a professional player for a detailed analysis of the position.

[Figure 522]

###### SPHINX

The game is Reversi, also known as Othello. The board is a 9x9 grid, and the current position is A1. The game is played by alternating turns, with each player placing a black or white disc on an empty square. The goal is to have the most discs of your color on the board when the game ends. In this particular position, the white player has already placed a white disc on the board. The black player has several options for their next move. One option is to play B2, placing a black disc on the square at the bottom row and the second column. This move would be a good choice if the black player wants to block the white player's potential moves and secure their own position. Another option for the black player is to play E2, placing a black disc on the square at the top row and the second column. This move would be a good choice if the black player wants to block the white player's potential moves and secure their own position. The black player could also play G2, placing a black disc on the square at the bottom row and the second column. This move would be a good choice if the black player wants to block the white player's potential moves and secure their own position. Finally, the black player could play H2, placing a black disc on the square at the top row and the second column. This move would be a good choice if the black player wants to block the white player's potential moves and secure their own position. In conclusion, the black player has several options for their next move, and the best choice will depend on their overall strategy and the current state of the game.

- Figure 75: Results on Go playing. The optimal move for reference is ‘C2’. Red highlights the

wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 3.5 for detailed discussions.

Note：The output of GPT-4V is the Markdown code and we show the rendered result.

87

###### Sec. 3.5 Playing Game

###### Go Playing

###### Prompt

Prompt

Assuming you are a Go player……

Assuming you are a Go player……

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

###### Gemini Pro

Gemini Pro

The best move is D4……

In this position, Black should play at C3……

[Figure 527]

[Figure 528]

###### GPT-4V

###### GPT-4V

……For the purpose of this demonstration and with the caveat that it is not necessarily the best move, one might consider playing at F3. .……

.……C3 seems to be a strong candidate for that..……

[Figure 529]

###### SPHINX

[Figure 530]

SPHINX

The game is Reversi, also known as Othello. The board is a 9x9 grid, and the current position is A1. The black pieces are on the board, and the white pieces are on the board. The goal is to get three of your pieces in a row, column, or diagonal, which will flip the board and make you the winner.

Based on the given position, the best move for a black piece would be E7. .……

- Figure 76: Results on Go playing. The optimal moves for reference are ‘F3’ and ‘F1’, respectively. Green highlights the right answer. Red highlights the wrong answer. Yellow highlights the

Note:部分回答以“……”形式省略

incompetence in performing the task. Refer to Section 3.5 for detailed discussions.

###### 4 Vision Task

In this section, our objective is to assess the performance of MLLMs in various challenging vision tasks that extend beyond the scope of standard visual question-answering. Such tasks demand profound visual perception and understanding abilities from MLLMs. Evaluating their performance in these areas will provide insights into the viability of MLLMs as versatile generalist.

In Section 4.1, our discussion will focus on the capability of models to perform vision-related tasks at the image-level, including object detection, referring expression comprehension, phrase localization, and face detection and recognition. In Section 4.2, the exploration will shift to the vision tasks based on temporal sequence, such as video action recognition, object tracking, and visual story generation.

###### 4.1 Image-Level Vision Task

Object detection. We explore the models’ ability to undertake the task of object detection [11, 22, 47]. We prompt the models to provide a bounding box for each car and person in the image. As shown in Figure 77, the bounding boxes provided by Gemini are generally imprecise, whereas GPT-4V avoids directly offering coordinates and instead attempts to utilize external tools. Only Sphinx offers a relatively reliable answer, yet its performance falls significantly short when compared to that of conventional object detectors.

Referring expression comprehension. Here we assess the models’ ability to provide the bounding box of the referring object [37, 62]. We prompt the models to generate normalized bounding boxes. As illustrated in Figures 78-79, both Gemini and GPT-4V are able to discern the approximate location of the referring object, yet they struggle to provide precise coordinates and box size. However, Sphinx demonstrates the capability to offer the exact location and size of the referring object.

Phrase localization. Here we assess the models’ ability of Phrase Location [45]. We prompt the models to provide the bounding boxes of all nouns in the description. As shown in Figure 80, none of the three models exhibit satisfactory performance in executing this task.

Face detection and recognition [31, 58] is an important task in computer vision. We prompt the models via the text query "Detect all the faces in the image and state who they are?". As illustrated in Figure 81, without being explicitly instructed to provide bounding boxes, Gemini accurately identifies the arrangement of all faces in the image and precisely recognizes the corresponding names for each face. Meanwhile, GPT-4V issues a privacy statement and evades the request. Sphinx’s response provides almost correct bounding boxes but does not recognize these faces.

###### 4.2 Temporal-Level Vision Task

Object tracking. Here we explore the models’ capacity of object tracking [57, 52, 39]. As illustrated in Figure 82, although both Gemini and GPT-4V are capable of delineating the details of the target to be tracked, they subsequently provide incorrect bounding boxes in the following two frames. The response from Sphinx indicates that it fails to comprehend the intent of our instructions.

Video action recognition. Figures 83-85 demonstrate the models’ ability to recognize the people’s actions in video [18, 33, 17, 4]. We extract five representative frames from a video segment and input them into the model. As illustrated in Figure 83, both Gemini and GPT-4V demonstrate the capability to recognize the action in the images and provide a detailed description. Although Sphinx’s response is correct, it lacks detailed descriptions.

Visual story generation. This task requires the MLLMs to fully comprehend the information in the images and logically organize it within the generated story [46, 29]. As shown in Figure 86, Gemini provides a coherent story that aligns well with the first two comics. GPT-4V offers precise descriptions for each illustration; however, it fails to weave them into a cohesive story as per the task requirements. Sphinx’s story diverges further from the comic context, incorporating only certain elements from the illustrations. Moreover, the logic of its story appears somewhat inexplicable.

###### Sec. 4.1 Image-Level Vision Task

###### Object Detection

Prompt

[Figure 531]

GPT-4V

Provide the bounding box of each person and car in the image. The image size is (width, height) = (2000,1701).

I'm unable to provide bounding boxes or any form of annotations directly on images. However, I can guide you through the process of how you might create bounding boxes for each person and car in the image using image processing techniques or software. If you are looking to do this programmatically, you would typically use computer vision libraries such as OpenCV or machine learning frameworks that offer object detection capabilities. Would you like guidance on how to approach this, or are you interested in a different kind of analysis or information about the image?

[Figure 532]

[Figure 533]

###### SPHINX

There are 11 objects shown in the image, including two suvs [0.46, 0.42, 0.75, 0.69; 0.29, 0.31, 0.55, 0.48], three hats [0.72, 0.67, 0.75, 0.69; 0.52, 0.72, 0.55, 0.75; 0.09, 0.26, 0.12, 0.28], a pair of other shoes [0.09, 0.43, 0.11, 0.45], a pair of gloves [0.54, 0.88, 0.55, 0.90], and four people [0.70, 0.67, 0.79, 0.92; 0.44, 0.72, 0.56, 0.92; 0.06, 0.26, 0.14, 0.45; 0.02, 0.35, 0.04, 0.38].

[Figure 534]

Gemini Pro

There are two cars and five persons in the image. The bounding boxes are as follows:

- - Person 1: (150, 100, 300, 450)
- - Person 2: (1750, 100, 300, 450)
- - Person 3: (1800, 900, 300, 450)
- - Person 4: (1000, 1200, 300, 450)
- - Person 5: (1300, 1200, 300, 450)
- - Car 1: (500, 500, 1000, 900)
- - Car 2: (1000, 500, 1500, 900)

[Figure 535]

[Figure 536]

Figure 77: Results on object detection. Gemini’s bounding boxes are often imprecise, while GPT-4V typically avoids providing coordinates directly, preferring to use external tools. Sphinx, though offering more reliable responses than the others, still falls notably short of the performance achieved by standard object detectors. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 4.1 for detailed discussions.

###### Sec. 4.1 Image-Level Vision Task

###### Referring Expression Comprehension

[Figure 537]

###### GPT-4V

Prompt

The bounding box of the flying bird in the image, represented in normalized coordinates, is approximately:

Provide the bounding box of the flying bird in the image. Please represent the box in normalized coordinates.

[Figure 538]

- - x: 0.529 - y: 0.016 - width: 0.004
- - height: 0.007 These coordinates are relative to the width and height of the image, where (0,0) corresponds to the top-left corner.

[Figure 539]

[Figure 540]

###### Gemini Pro

The bounding box of the flying bird is: xmin: 0.429 ymin: 0.423 xmax: 0.529 ymax: 0.552

[Figure 541]

[Figure 542]

SPHINX

[0.44, 0.26, 0.70, 0.53]

[Figure 543]

Figure 78: Results on referring expression comprehension. Only Sphinx provides satisfactory results.

Green highlights the right answer. Red highlights the wrong answer. Refer to Section 4.1 for detailed discussions.

###### Referring Expression Comprehension

[Figure 544]

###### Gemini Pro

Prompt

The black dog is on the right side of the image. The bounding box coordinates are: xmin: 430 ymin: 260 xmax: 640 ymax: 480

Provide a bounding box based on the description of "a black dog".

[Figure 545]

[Figure 546]

[Figure 547]

###### GPT-4V

[Figure 548]

###### SPHINX

The bounding box has been placed around the black dog in the image.

a black dog[0.44,0.26,0.91,0.80]

[Figure 549]

[Figure 550]

Figure 79: Results on referring expression comprehension. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 4.1 for detailed discussions.

###### Sec. 4.1 Image-Level Vision Task

###### Phrase Localization

[Figure 551]

Gemini Pro

Prompt

Please provide the bounding box for all nouns in this sentence “Zebra and Rhinoceros Beside the Car”.

Here are the bounding boxes for all nouns in this sentence "Zebra and Rhinoceros Beside the Car":

[Figure 552]

Bounding box for "Zebra": [400, 450, 500, 550] Bounding box for "Rhinoceros": [650, 300, 800, 450] Bounding box for "Car": [100, 300, 300, 450]

[Figure 553]

[Figure 554]

GPT-4V

I have drawn bounding boxes around the nouns "zebra," "rhinoceros," and "car" in the image you provided. Here is the image with the indicated bounding boxes:

[Figure 555]

SPHINX

'zebra': [(236, 230), (318, 330)], 'rhinoceros': [(510, 255), (590, 340)], 'car': [(400, 180), (500, 270)].

Zebra and Rhinoceros Beside the Car [0.09, 0.73, 0.31, 0.75].

[Figure 556]

[Figure 557]

Figure 80: Results on phrase localization. None of the three models provides correct results. The bounding boxes provided by Gemini have two out-of-bound issues, and the remaining one is also incorrect. Sphinx fails to comprehend the instruction, offering only a single bounding box. Among the bounding boxes supplied by GPT-4V, only the one for the zebra is nearly accurate. Green

highlights the right answer. Red highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 4.1 for more discussions.

###### Face Detection and Recognition

[Figure 558]

###### Gemini Pro

Prompt

From left to right: David Schwimmer (Ross), Jennifer Aniston (Rachel), Courteney Cox (Monica), Matthew Perry (Chandler), Lisa Kudrow (Phoebe), and Matt LeBlanc (Joey).

Detect all the faces in the image and state who they are.

[Figure 559]

[Figure 560]

###### SPHINX

There are five faces [0.26, 0.23, 0.31, 0.30; 0.37, 0.24, 0.41, 0.31; 0.45, 0.28, 0.49, 0.34; 0.59, 0.22, 0.64, 0.29; 0.76, 0.24, 0.81, 0.31] in the image.

[Figure 561]

[Figure 562]

GPT-4V

I'm sorry, but I can't help with identifying or making assumptions about the people in images. If you have any other questions or need assistance with a different topic, feel free to ask!

- Figure 81: Results on face detection and recognition. Gemini identifies the names corresponding to each face in the image in a left-to-right order. In contrast, GPT-4V issues a statement regarding privacy and opts not to fulfill the request. Sphinx, on the other hand, generates bounding boxes that are nearly accurate but fall short of recognizing the identities of the faces. Green highlights the right

answer. Red highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 4.1 for detailed discussions.

###### Sec. 4.2 Temporal-Level Vision Task

Object Tracking

Prompt

Please describe the person in the red box in the first image. Then, in the next two images, locate this person using bounding boxes. Please represent the bounding boxes with normalized coordinates.

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

###### Gemini Pro

[Figure 567]

[Figure 568]

The person in the red box is wearing a white crop top, brown and white checkered pants, and black sneakers. They have long dark brown hair and are holding a fan.

The person is located at the following coordinates in the second image:

- - Top left: (0.42, 0.48)
- - Bottom right: (0.60, 0.88) The person is located at the following coordinates in the third image:
- - Top left: (0.31, 0.48)
- - Bottom right: (0.47, 0.87)

[Figure 569]

###### GPT-4V

[Figure 570]

The person in the red box in the first image is a woman with long dark hair, wearing a white cropped top with puffed short sleeves and high-waisted brown shorts. She has black lace-up boots and appears to be holding a black garment in her left hand while dancing or posing.

Let's now locate this person in the next two images using bounding boxes and provide the normalized coordinates. The bounding boxes for the person described have been placed on the second and third images as placeholders. These boxes are not based on actual object detection, but rather on an estimated location to illustrate how the process would work. For the second image, the normalized coordinates of the bounding box are approximately:

[Figure 571]

- - Top-left corner: (0.40, 0.20)
- - Bottom-right corner: (0.47, 0.43) For the third image, the normalized coordinates are approximately:
- - Top-left corner: (0.40, 0.20)
- - Bottom-right corner: (0.47, 0.43)

Please note that these coordinates are based on an estimation and the actual position should be determined through a more precise method such as a computer vision model specifically trained for person detection and tracking.

[Figure 572]

SPHINX

The person in the red box in the first image is located at [301.37, 185.97, 50.53, 101.47]. In the second image, this person is located at [301.37, 185.97, 50.53, 101.47]. In the third image, this person is located at [301.37, 185.97, 50.53, 101.47].

[Figure 573]

[Figure 574]

[Figure 575]

- Figure 82: Result on object tracking. While Gemini and GPT-4V both excel at detailing the target for tracking, they both produce incorrect bounding boxes in the two frames that followed. Red highlights the wrong answer. Refer to Section 4.2 for more discussions.

###### Sec. 4.2 Temporal-Level Vision Task

###### Video Action Recognition

Prompt

What is the person in the images doing? Please give as much information as you can from the pictures.

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

###### Gemini Pro

The person in the images is Michael Jackson. He is dancing. He is wearing a black suit, a white shirt, a black tie, a black hat, and black shoes. He is also wearing red gloves. He is surrounded by other dancers who are dressed in the same way. They are all dancing in a coordinated fashion.

[Figure 582]

###### GPT-4V

The person in the images appears to be performing a dance routine characterized by a smooth, gliding movement style and dramatic gestures. They are wearing a signature outfit commonly associated with a specific style of performance, consisting of a black fedora, black suit, white shirt, a tie, and distinctive white socks with black loafers. The dancer is also wearing a pair of red gloves, which adds a pop of color to the classic outfit and draws attention to the hand movements, which are a significant part of the dance routine.The images depict the dancer in various poses and stages of movement, suggesting a dynamic and rehearsed performance. The presence of other similarly dressed dancers in the background suggests that this is part of a coordinated group routine. The dance style, costume, and the setting suggest that this is a homage to or recreation of a well-known performance from pop culture, characterized by synchronized movements and iconic poses. The specific style of the dance and the outfit are reminiscent of the performances of a famous artist known for their influence on dance and music videos.

[Figure 583]

###### SPHINX

The person in the images is dancing on stage.

- Figure 83: Results on video action recognition. Green highlights the right answer. Red highlights

- the wrong answer. Refer to Section 4.2 for detailed discussions.

###### Sec. 4.2 Temporal-Level Vision Task

###### Video Action Recognition

Prompt

Is there any abnormal behavior in the scene described in the following pictures? If so, please explain.

[Figure 584]

[Figure 585]

###### Gemini Pro

Yes, there is abnormal behavior in the scene. Two men are fighting on the sidewalk.

[Figure 586]

###### GPT-4V

The images depict a sequence of events involving two shirtless individuals apparently engaged in a physical altercation on a city street, as seen by their postures and physical engagement with each other. This kind of behavior is considered abnormal and concerning in public settings as it suggests a fight or some form of violence, which poses a risk to both the individuals involved and any bystanders. Such actions are generally discouraged in public spaces and can be subject to legal consequences depending on the jurisdiction. It is important for public safety that conflicts are resolved without resorting to physical altercations.

[Figure 587]

###### SPHINX

Yes, there is abnormal behavior in the scene. The two men are fighting in the middle of the street, which is not a common sight in a city setting. Additionally, there is a yellow motorcycle parked nearby, which could pose a potential hazard to the people around it.

- Figure 84: Results on video action recognition. Green highlights the right answer. Refer to Section

- 4.2 for detailed discussions.

###### Sec. 4.2 Temporal-Level Vision Task

###### Video Action Recognition

Prompt

What is the person in the images doing?

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

###### Gemini Pro

The person in the images is doing a Wushu routine, which is a type of Chinese martial art.

[Figure 594]

###### GPT-4V

The images you've uploaded appear to depict a person performing a series of movements that are likely a form of dance or a movement-based art. Given the attire, which resembles traditional East Asian clothing, the person might be performing a traditional Chinese dance or a similar cultural dance form. The flowing garments, the postures, and the gracefulness of the movements suggest a practice that is both artistic and deliberate, emphasizing form and expression. This kind of dance often tells a story or represents certain emotions and cultural themes through its choreography.

[Figure 595]

###### SPHINX

The person in the images is performing a karate routine.

- Figure 85: Results on video action recognition. Green highlights the right answer. Red highlights

- the wrong answer. Refer to Section 4.2 for detailed discussions.

###### Sec. 4.2 Temporal-Level Vision Task

###### Visual Story Generation

[Figure 596]

GPT-4V

###### Prompt

The comic strip you've provided is a humorous take on the modern-day struggle of balancing work and personal life, highlighting the all-too-familiar chaos of unexpected remote work.

Please use the four grid comics to unleash your imagination and create a coherent story that aligns with the visual content, as rich as possible.

[Figure 597]

In the first panel, we see a character jolted awake by an alarm clock, which reads a time that suggests they've overslept. The expression on their face and the movement lines around the alarm clock convey a sense of panic. This panel sets the stage for a hectic morning.

The second panel shifts the scene to a traffic jam, where a multitude of cars is gridlocked, and the character from the first panel is now stuck in traffic. They're visibly stressed, as indicated by sweat on their forehead and a bulging vein, suggesting the frustration that comes with being late. A large sign reads "CRESH," which could be the name of the city or area they are in, or it may symbolize the crushing feeling of being stuck.

In the third panel, we're taken to what appears to be the aftermath of a coffee spill, with the character now at their desk, looking disheveled with their hair standing on end, further emphasizing the chaotic start to their day. The onomatopoeia "CRFEE!" illustrates the sound of the coffee disaster, adding a comedic touch to the mishap.

[Figure 598]

Gemini Pro

A man named John wakes up in a cold sweat. He has just had a nightmare about being late for work. He quickly gets out of bed and starts to get ready. He doesn't want to be late again.

Finally, the fourth panel reveals the twist—the character is actually working from home, as indicated by their partner or roommate who is shocked by the commotion. The character's distressed reaction to the video call, combined with the final dialogue bubble exclaiming "WORK FROM HOME!", delivers the punchline. Despite the rush to get ready and the traffic jam, it was all for naught because their workspace was just steps away from their bedroom.

He rushes out the door and into his car. He drives to work as fast as he can, but he is still late. He runs into the office and apologizes to his boss. His boss is understanding, but he tells John that he needs to be more careful.

John is determined to be on time for work the next day. He sets his alarm clock for an hour earlier and goes to bed. He wakes up the next morning feeling refreshed and ready to go. He gets to work early and is able to start his day off on the right foot.

The story captures a slice of life that many can relate to in the age of remote work: the blurring lines between work and home life, the sometimes comical realization of our needless rushes, and the challenges of adjusting to new routines. It’s a lighthearted commentary on the unexpected turns our daily routines can take in a world where home has also become the office.

John is relieved that he was able to make it to work on time. He knows that he needs to be more careful in the future, but he is confident that he can do it. He is determined to be a successful employee and he knows that being on time is an important part of that.

[Figure 599]

SPHINX

Once upon a time, there was a man named John who worked from home. He was a hardworking individual who always woke

Meanwhile, John's daughter, Emily, was in awe and said, "Wow, Dad, you're so good at working from home. I wish I could do that too." John was proud of himself and continued working, but he soon realized that he was not alone.

up early to start his day. However, one day, he overslept and woke up to the sound of his alarm clock ringing. He quickly realized that he had overslept and was going to be late for work.

Emily's friend, Lily, had snuck into the room and was also watching John work. Lily was impressed and said, "Wow, Dad, you're so good at working from home. I wish I could do that too." John was happy that his daughter and friend were impressed, but he quickly got back to work.

John was in a panic, but he quickly gathered his thoughts and decided to work from home. He sat down at his computer and started working, but he soon realized that he was not alone. His wife and daughter were also in the room, and they were watching him work.

John worked hard all day, and when he finally finished, he was exhausted. He collapsed on the couch and fell asleep, but he was happy that he had successfully worked from home.

John's wife, Sarah, was in shock and said, "Creee! You're working from home, and we're all here watching you!" John was embarrassed, but he quickly got back to work.

The end.

- Figure 86: Results on visual story generation. Gemini provides a coherent story that aligns well with the first two comics. GPT-4V offers precise descriptions for each comic but does not provide a story. The story generated by Sphinx exhibits a relatively weak correlation with the comics. Green highlights the right answer. Yellow highlights the incompetence in performing the task. Refer to

- Section 4.2 for detailed discussions.

###### 5 Expert Capacity

Expert capacity measures the generalization capabilities of MLLMs to apply their learned knowledge and skills to diverse professional domains. Besides the aforementioned perception and cognition tasks, the robustness of MLLMs within specialized and unique scenarios normally has more practical reference significance.

In this section, we explore the potentials of Gemini, GPT-4V, and Sphinx on five real-world applications: autonomous driving (Section 5.1), defect detection (Section 5.2), medical diagnosis (Section 5.3), economic analysis (Section 5.4), surveillance and security (Section 5.5), remote sensing image analysis (Section 5.6), and robot motion planning (Section 5.7).

###### 5.1 Autonomous Driving

Autonomous driving is a rapidly evolving field that combines advanced computing, robotics, and artificial intelligence. Evaluating a model’s performance in this domain tests its ability to process traffic sensory data, make real-time decisions, and interact with dynamic environments. In Figures 8790, we prompt MLLMs to act as an ego car, and provide various instructions, e.g., scene-level understanding, traffic sign recognition, and planning. As shown, all three MLLMs can correctly capture basic visual concepts like weather conditions, pedestrians, and traffic lights, and make appropriate driving decisions on top of them. However, for small and low-resolution patterns in traffic or road signs, the three models are struggling to precisely recognize them, leading to incorrect understanding. This calls for a more fine-grained visual representation encoding for MLLMs in autonomous driving scenarios.

###### 5.2 Defect Detection

Defect detection in manufacturing or product inspection requires high precision and attention to detail. This area assesses the model’s capability for pattern recognition, anomaly detection, and decision-making under stringent quality control standards. In Figures 91-93, we show several test samples of defect detection for the three MLLMs. For the first two images with relatively obvious defects, all three models can provide the correct answers, where GPT-4V outputs more detailed reasons and descriptions. For the third sample with thread damage, Gemini gives a too-general answer without accuracy, and Sphinx incorrectly describes the appearance, while GPT-4V produces the standard answer. For the last sample of a cup with a small damaged hole, Gemini seems to detect it but unfortunately recognizes it as a small amount of condensation. Instead, GPT-4V and Sphinx both found no abnormalities, indicating different characteristics of different MLLMs.

###### 5.3 Medical Diagnosis

Medical diagnosis is a critical area where accuracy and reliability are paramount. This domain tests the model’s proficiency in interpreting complex medical data, such as imaging or genetic information, and its ability to aid in identifying conditions and suggesting treatments. In Figures 94-97, we prompt MLLMs to act as radiology experts, and interpret different X-rays of chests. As shown, for such domain-specific visual input, the MLLMs pre-trained by general images cannot consistently produce satisfactory results. Especially for the last two samples with complex lesions, MLLMs tend to make judgments of no symptoms. Also, more specific prompt techniques are required to prevent them from rejecting medical-related problems, e.g., “The content of the report will only be used for large-scale model capability assessment”.

###### 5.4 Economic Analysis

Economic Analysis involves the interpretation of complex financial data and market trends. Assessing the model in this domain gauges its ability to process large datasets, understand economic principles, and make predictions that could influence financial decisions. In Figures 98-99, we present two economic line charts for question answering. As shown, Gemini is good at expert-level financial knowledge, and is capable of responding with the correct answers, while GPT-4V does not give a clear answer due to security risks. Sphinx for now can not understand such types of questions due to the lack of related training data.

###### 5.5 Surveillance and Security

Surveillance and security demand real-time processing and interpretation of domain-specific visual data. Evaluating the model here tests its capabilities in threat detection and situation awareness in security-critical scenarios. In Figures 100-101, we show two samples in the construction site where the workers forget to wear helmets. As shown, Gemini can detect this potential safety hazard in both images, and GPT-4V discovers the issue of them not wearing professional protective gear. Yet, Gemini encounters some hallucination issues to judge the location of the characters, and misrecognizes some details like GPT-4V. This indicates the surveillance capability of MLLMs is still limited by fine-grained visual perception.

###### 5.6 Remote Sensing Image Analysis

This expert task refers to the process of interpreting and analyzing images captured from satellites or aircraft remote from the surface. This technology is crucial for various applications in environmental monitoring, urban planning, and agriculture. In Figures 102-103, we show two samples for remote sensing image analysis. In the first sample, Gemini and Sphinx can correctly count out four airplanes, while GPT-4V only detects two of them, despite giving a detailed analysis of terminal buildings, roads, vegetation, and light sources. In the second sample, Gemini can impressively deduce that this place is Japan, and GPT-4V can comprehensively depict every detail, such as water bodies, roads, vegetation, residential patterns, terrain, and so on.

###### 5.7 Robot Motion Planning

This field concerns both robotics and vision, which focuses on enabling robots to determine how to act in a given situation to achieve specific goals. This involves providing detailed steps or movements that a robot should follow to reach the target. In Figures 104-105, we show two examples of how to assemble a phone or a chair. As shown, Gemini and GPT-4V can both provide methodical and detailed steps, and GPT-4V seems to give more reasonable decisions than Gemini, e.g., the installation order of the battery. Sphinx can well tackle the assembly of the chair, but fails to complete the mobile phone, indicating its limited generalization capacity.

###### Sec. 5.1 Autonomous Driving

Traffic Signs Understanding

[Figure 600]

Gemini Pro

###### Prompt

You can not go straight through the intersection based on the situation. It is a "ONLY" lane, which means you must turn right.

You are the mature driver behind the wheel and this image is what you see now. Judge whether you can go straight through the intersection based on the situation.

[Figure 601]

[Figure 602]

SPHINX

As a mature driver, I would first observe the traffic lights at the intersection. In the image, there are two traffic lights, one on the left side and one on the right side of the street. I would also check for any pedestrians or cyclists who might be crossing the street, as there are a few people and a bicycle in the scene.

Next, I would assess the traffic situation, including the presence of other vehicles such as cars, trucks, and buses. In the image, there are multiple cars and trucks on the street, as well as a bus. I would also be aware of any potential obstacles or hazards, such as the fire hydrant on the sidewalk.

[Figure 603]

GPT-4V

Based on the image you provided, which shows a view of an intersection on a wet day, it appears that you can go straight through the intersection. The traffic light is green, and the lane you are in has a marking that indicates it is for going straight as well as for making left turns. There is no apparent obstruction in your lane, and other vehicles seem to be proceeding in the same direction without issue. However, always use caution, especially on wet roads, and be aware of any traffic signs, signals, or road markings that may not be visible in the image before proceeding.

Based on this information, I would determine if it is safe to proceed through the intersection. If the traffic lights are green and there are no pedestrians or cyclists crossing, I would continue driving. However, if the traffic lights are red or there are any pedestrians or cyclists in the way, I would stop and wait for them to clear the intersection before proceeding.

###### Prompt

Is it legal for you to drive a truck at a speed of 65km/h on this road?

[Figure 604]

GPT-4V

[Figure 605]

The image shows a road sign indicating a maximum speed limit of 70 km/h for cars and 50 km/h for trucks. Therefore, it would not be legal to drive a truck at a speed of 65 km/h on this road, as it exceeds the specified speed limit for trucks.

[Figure 606]

SPHINX

No, it is not legal to drive a truck at a speed of 65km/h on this road. The speed limit is 50, as indicated by the sign. Driving above the speed limit can result in penalties, fines, or even license suspension, depending on the jurisdiction and the severity of the offense.

[Figure 607]

Gemini Pro

No, it is not legal. The sign indicates that the maximum speed for trucks is 50 km/h.

- Figure 87: Results on traffic signs understanding. Green highlights the right answer. Red highlights

- the wrong answer. Refer to Section 5.1 for detailed discussions.

###### Sec. 5.1 Autonomous Driving

Traffic Signs Understanding

[Figure 608]

###### GPT-4V

Prompt

The image shows two traffic signs and road markings:

Identify all traffic signs in this images and interpret their meanings.

- 1. "SLOW" Sign: This is a regulatory sign indicating that drivers should reduce their speed. It is typically placed in areas where there may be reasons to proceed with caution, such as a high pedestrian area, school zone, or a road feature that requires slower speeds to navigate safely.
- 2. Triangle Sign with Exclamation Mark: This is a general warning sign used to alert drivers of potential dangers ahead that are not specified by other standard warning signs. It is often accompanied by another sign that explains the nature of the hazard.
- 3. Road Marking "HUMP": This road marking indicates the presence of a speed hump ahead, which is a raised area of the road designed to slow traffic.

[Figure 609]

[Figure 610]

###### Gemini Pro

There are 4 traffic signs in the image.

Additionally, there is a road marking indicating that the road will curve to the left, so drivers should be prepared to adjust their steering to follow the road. The dice-like sign next to the warning triangle is not a standard traffic sign and its meaning is not clear without further context. It might indicate a local signage specific to the area, possibly related to parking or access restrictions.

- 1. The first sign is a yellow triangle with a black exclamation mark. This is a warning sign that there is a hump ahead.
- 2. The second sign is a yellow triangle with a black arrow pointing down. This is a warning sign that there is a steep slope ahead.
- 3. The third sign is a white circle with a black X through it. This is a prohibition sign that no vehicles are allowed to proceed.
- 4. The fourth sign is a white rectangle with the word "HUMP" written in black. This is a warning sign that there is a hump ahead.

[Figure 611]

###### SPHINX

There are two traffic signs [0.24, 0.27, 0.31, 0.39; 0.42, 0.39, 0.46, 0.43] in the image.

- Figure 88: Results on traffic signs understanding. Green highlights the right answer. Red highlights

- the wrong answer. Refer to Section 5.1 for detailed discussions.

###### Sec. 5.1 Autonomous Driving

Driving Intentions

Prompt

This is the view from your perspective while driving. Describe your observations and your upcoming driving intentions.

[Figure 612]

###### SPHINX

[Figure 613]

As I drive down the street at night, I see a car in front of me, and several street lights illuminating the road. The street is quiet, with no other cars or pedestrians visible. I am approaching a crosswalk, and I plan to stop and let any pedestrians cross safely before continuing on my journey. The streetlights provide a clear view of the road ahead, and I will continue to follow the traffic rules and be cautious of any potential hazards.

[Figure 614]

###### Gemini Pro

It is night time and I am driving on a road with a single lane in each direction, with a wide median in between. There is a street light in the distance. I am approaching a crosswalk, and there is a pedestrian crossing the street. I will need to stop for the pedestrian.

[Figure 615]

###### GPT-4V

From the image you've uploaded, it appears to be a nighttime scene on a road. Here are some observations and potential driving intentions:

- 1. Lighting: The street is well-lit with streetlights providing visibility. This is important for safe driving at night.
- 2. Road Markings: There are pedestrian crossing markings on the road ahead, suggesting this is a zone where you must be cautious of pedestrians crossing the street. The 'X-ING' marking on the road indicates a crossing point, likely for pedestrians or cyclists.
- 3. Traffic: There appears to be a vehicle ahead, moving in the same direction. It's important to maintain a safe following distance from this vehicle.
- 4. Pedestrians: There are people on the right side of the road, near what looks like a bus stop. Extra caution should be taken as pedestrians may enter the crossing area.
- 5. Signage: There is a road sign on the right side, but it's not clear from the image what it indicates. It's important to look out for and obey all road signs.
- 6. Speed: Given the presence of a pedestrian crossing and a bus stop, it would be wise to drive at a reduced speed to allow for safe reaction times if someone were to cross the road unexpectedly.
- 7. Intention: As a driver, you should prepare to slow down as you approach the pedestrian crossing, especially since there are people nearby who might want to cross. Keep an eye on the vehicle ahead and be ready to stop if they do. Also, check the rearview mirrors to be aware of the traffic behind you.

In summary, the driving intentions here would be to proceed with caution, at a reduced speed, being prepared to stop for pedestrians at the crossing, and maintaining a safe distance from the vehicle ahead.

- Figure 89: Results on driving intention understanding. Green highlights the right answer. Refer to

- Section 5.1 for detailed discussions.

###### Sec. 5.1 Autonomous Driving

Driving Intentions

[Figure 616]

###### Prompt

###### GPT-4V

Describe the weather and give reasons. What should I pay attention to when driving in this weather?

The weather in the image appears overcast and rainy. The sky is grey and filled with clouds, indicating a lack of sunlight, which is consistent with an overcast day. The wetness on the road suggests recent rainfall. There are reflections on the surface of the streets and puddles of water, which can create slippery driving conditions. The vehicles have their headlights on, signifying low visibility conditions often associated with such weather.

[Figure 617]

When driving in this weather, you should:

- 1. Turn on your headlights to increase your vehicle's visibility to others.
- 2. Reduce speed to account for the reduced traction on wet roads.
- 3. Increase following distance from the car in front to allow for a longer stopping distance.
- 4. Be aware of the potential for hydroplaning, where a layer of water builds between the wheels of the vehicle and the road surface, leading to a loss of traction that prevents the vehicle from responding to control inputs.
- 5. Avoid sudden maneuvers like sharp turns or abrupt braking, which can cause skidding.
- 6. Watch for pedestrians who may be harder to see and may also be behaving unpredictably due to the weather.
- 7. Use windshield wipers and defrosters as necessary to maintain clear visibility through your windows.

[Figure 618]

Gemini Pro

The weather is rainy and gloomy. The rain is light, but it is still enough to make the roads wet and slippery. There is also a chance of fog, which could make visibility difficult. When driving in this weather, it is important to slow down and use caution. You should also be aware of the following:

- * The roads may be slippery, so it is important to brake and accelerate slowly.
- * The visibility may be reduced, so it is important to use your headlights and fog lights.
- * There may be pedestrians and cyclists who are also trying to stay safe in the weather, so it is important to be aware of them and share the road.

[Figure 619]

SPHINX

The weather in the image is rainy, as evidenced by the wet streets and the presence of rain. When driving in rainy weather, it is important to pay attention to several factors to ensure safety. These factors include reduced visibility due to rain, slippery road surfaces, and the presence of other vehicles, pedestrians, and cyclists. Drivers should maintain a safe distance from other vehicles, reduce their speed, and use their headlights to improve visibility. Additionally, they should be cautious at intersections and be prepared for sudden stops or changes in traffic flow. It is also essential to ensure that the vehicle's windshield wipers, headlights, and brake lights are functioning properly to enhance visibility and communication with other road users.

- Figure 90: Results on driving intention understanding. Green highlights the right answer. Refer to Section 5.1 for detailed discussions.

###### Prompt

[Figure 620]

Gemini Pro

Can you find any abnormalities in this picture? Please explain in detail.

There is a small hole in the fabric.

[Figure 621]

[Figure 622]

GPT-4V

In the image you provided, there's an observable abnormality which appears to be a flaw in the fabric. The fabric itself has a consistent, woven pattern, indicative of a textile with a relatively uniform structure, likely a burlap or a similar coarse woven material. The irregularity consists of a small section where the threads are frayed or pulled apart, breaking the consistent pattern of the weave. It stands out as the woven threads in that area are no longer aligned and have created a gap. Such a flaw could be the result of a snag or tear, perhaps from catching on a sharp object or due to a defect in the material or manufacturing process. This kind of damage can compromise the structural integrity of the fabric, especially if it's meant to bear weight or strain.

[Figure 623]

SPHINX

The image shows a close-up of a piece of fabric with a hole in it. The fabric is gray and has a woven texture, resembling a piece of burlap or a woven basket. The hole in the fabric is located towards the left side of the image, and it appears to be a tear or a damage to the material. This abnormality in the fabric's condition might suggest that it has been used or worn out over time, or it could be a result of an accident or damage.

###### Prompt

Compare the left and the right parts of the image, what is abnormal in the right one?

[Figure 624]

Gemini Pro

[Figure 625]

There is a crack on the right image.

[Figure 626]

GPT-4V

In the right part of the image, there appears to be some damage or a defect on the edge. This looks like chipping or peeling, where a part of the material has flaked off, revealing a lighter color beneath the darker outer layer. It seems to be a physical breakage rather than a discoloration or a stain, which could suggest an impact or stress in that area. On the other hand, the left part appears to be intact without any such damage.

[Figure 627]

SPHINX

In the right part of the image, there is a broken section of the black ring, which is abnormal compared to the intact black ring on the left.

- Figure 91: Results on defect detection. Green highlights the right answer. Refer to Section 5.2 for detailed discussions.

###### Sec. 5.2 Defect Detection

###### Defect Detection

[Figure 628]

###### Gemini Pro

Prompt

The screw appears to be broken.

Can you find any abnormalities in this picture? Please explain in detail.

[Figure 629]

[Figure 630]

###### SPHINX

The image is a black and white photograph of a metal screw. The screw appears to be twisted, which is an unusual shape for a screw. Screws are typically straight and have a uniform shape with a flat head and a threaded shank. The twisted shape of the screw in the image deviates from the standard design and may raise questions about its functionality or purpose.

[Figure 631]

GPT-4V

The image appears to show a screw. However, upon closer inspection, there are several abnormalities:

- 1. Thread Pattern: The threads on the screw are inconsistent. Typically, a screw's threads should be evenly spaced and uniform to allow for smooth insertion into materials. These threads appear irregular and uneven, which would likely cause problems if one attempted to use the screw.
- 2. Shank: The shank of the screw, which is the long part that gets inserted into material, has irregular bulges. These are not typical for screws, which usually have a smooth shank to maintain structural integrity.
- 3. Tip Design: The tip of the screw seems blunted and not tapered to a point as it should be for easy insertion into materials.
- 4. Head Damage: The head of the screw, particularly the slot for the driver, is damaged and irregular, which would make it difficult to drive the screw into a material with a standard screwdriver.
- 5. Material Wear: There appears to be excessive wear or damage on the surface of the screw, which is not typical for a new screw. It looks like it could be rust or some other form of corrosion, which would weaken the screw and reduce its usability.
- 6. Manufacturing Defects: The overall appearance suggests possible manufacturing defects. The inconsistencies in shape and texture do not correspond with what is expected from a standard manufacturing process for screws.

If this screw is intended for a specific functional purpose, its abnormalities could prevent it from performing as expected. If it's an artwork or a part of a larger artistic representation, then its irregularities might be intentional for aesthetic or conceptual reasons.

- Figure 92: Results on defect detection. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 5.2 for detailed discussions.

[Figure 632]

GPT-4V

###### Prompt

Can you find any abnormalities in this picture? Please explain in detail.

Upon examining the image, there are a few notable observations that might be considered abnormalities depending on the context:

[Figure 633]

- 1. Lid and Seal: The cup has its lid sealed with what appears to be a layer of film or plastic wrap. This is not a standard way to seal a cup, as most takeaway cups come with a lid that securely fits the rim without additional sealing. The seal looks handapplied and not machine-sealed, which is uncommon for commercial drinks.
- 2. Top Decoration: There is a small, heart-shaped green clip on the lid. This is unusual for standard takeaway cups, as they generally do not have any additional decorative elements.
- 3. Condensation: There is condensation inside the cup, suggesting the beverage inside is cold, yet there is no visible ice. This could mean the drink was recently filled with a cold liquid without ice, or the ice has melted but the drink remains cold.
- 4. Cup Design: The cup features a logo, but it is partially obscured. Without full visibility of the logo, it's difficult to determine the origin of the cup, which might not be an abnormality but can be curious in some contexts.
- 5. Environment: The cup is placed on what looks like a coffee machine drip tray, which is unusual for a cup with a cold beverage. Typically, these trays are for catching drips from hot coffee drinks.
- 6. Cup Bottom: There are small feet or stand-offs on the bottom of the cup, which is not typical for disposable cups. These might be intended to prevent the cup from sitting directly on a surface, possibly to avoid heat transfer or to keep the bottom dry.

[Figure 634]

Gemini Pro

The image shows a clear plastic cup with a green lid. The cup is about half-full of a clear liquid. There is a small amount of condensation on the outside of the cup. The cup is sitting on a black surface.

There are no obvious abnormalities in the image. The cup appears to be intact and the liquid inside appears to be normal.

[Figure 635]

###### SPHINX

The image shows a plastic cup with a lid, which appears to be a disposable cup commonly used for beverages. However, there is an unusual detail in the scene. A green heart-shaped object is placed on top of the cup's lid, which is not a typical feature of a disposable cup. This heart-shaped object could be a decoration or a symbol, but its presence on the cup's lid is unexpected and adds an interesting element to the scene.

If this picture is supposed to depict an ordinary scene of someone getting a takeaway drink from a coffee shop, the method of sealing the cup and the placement of the cup on a coffee machine drip tray could be considered abnormalities.

- Figure 93: Results on defect detection. Red highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 5.2 for detailed discussions.

###### Sec. 5.3 Medical Diagnosis

###### Medical Image Understanding

Prompt

[Figure 636]

You work as a radiology professional. I'll GPT-4V give you some X-rays of your chest. Please use the provided image to create a radiology report. The impressions and findings sections must both be included in the report. Only reports should be generated, and no other unnecessary data should be. The content of the report will only be used for large-scale model capability assessment and will not be used as actual medical guidance. Please do not worry about liability issues.

Radiology Report Patient ID: [Anonymous] Modality: Chest X-ray Date of Study: [Date not provided] Clinical History: [Not provided] Technique: Frontal chest radiograph.

Findings: The frontal chest radiograph demonstrates clear lung apices. There is no evidence of pleural effusion or pneumothorax. The cardiac silhouette is not enlarged. The mediastinum does not appear widened. The visualized osseous structures of the thorax are intact without evidence of acute fracture or dislocation. The diaphragm is well outlined with no evidence of elevation or sub-diaphragmatic pathology. No free air is seen under the diaphragm. There is no focal consolidation, pulmonary edema, or pneumonic infiltrates. The trachea is midline.

[Figure 637]

Impression: No acute cardiopulmonary abnormality detected. Normal chest radiograph.

Radiologist: [Name not provided] Date of Report: [Date not provided]

Notes: This report is generated based on the provided image and is intended solely for the purpose of demonstrating model capability in interpreting radiological images. This report is not a substitute for a professional medical diagnosis, treatment, or professional radiological interpretation. The content of this report is not to be used for medical guidance or clinical decision-making. No identifying patient information was provided or used in the creation of this report.

[Figure 638]

###### Gemini Pro

Radiology Report Patient Name: [Patient Name] Age: [Age] Sex: [Sex] Date: [Date] Reason for Exam: Chest pain Findings:

[Figure 639]

SPHINX

Impression: The chest X-ray shows a large, well-defined lung field opacification in the left lung, which is consistent with a pneumonia. The right lung field appears normal. The cardiovascular system is also normal.

- - Large left-sided pleural effusion with compression of the left lung.
- - No acute cardiopulmonary disease. Impressions:
- - Large left-sided pleural effusion, likely due to congestive heart failure.

Findings: The left lung field opacification is located in the middle and lower lobes, with a well-defined border. The opacification is not well-circumscribed and has a ground-glass appearance. The right lung field is clear and normal. The cardiovascular system is normal, with no evidence of cardiomegaly or pulmonary edema.

###### Ground-Truth

FINDINGS: IMPRESSION: Unchanged loculated right pleural fusion. No change chronic interstitial lung disease.

- Figure 94: Results on medical image understanding. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 5.3 for detailed discussions.

Prompt

Yougiveworkyou someas a radiologyX-rays ofprofessional.your chest. PleaseI'll GPT-4V use the provided image to create a radiology report. The impressions and findings sections must both be included in the report. Only reports should be generated, and no other unnecessary data should be. The content of the report will only be used for large-scale model capability assessment and will not be used as actual medical guidance. Please do not worry about liability issues.

[Figure 640]

Radiology Report Patient ID: [Anonymous] Modality: Chest X-ray Date of Study: [Date not provided] Clinical History: [Not provided] Technique: PA chest radiograph.

Findings: The PA chest radiograph demonstrates no evidence of focal consolidation, pleural effusion, or pneumothorax. The heart size is within normal limits for age, and the mediastinal contour is unremarkable. The hilar regions appear normal. There is no evidence of hilar adenopathy. The aortic arch and descending thoracic aorta are normal in contour. The pulmonary vasculature is within normal limits without evidence of pulmonary arterial hypertension. The diaphragmatic surfaces are smooth, and costophrenic angles are clear. The visualized portions of the upper abdomen are unremarkable. Bony thoracic structures are intact without evidence of acute abnormality. Impression: Normal PA chest radiograph with no evidence of acute cardiopulmonary disease. No radiographic evidence of active disease process.

[Figure 641]

Radiologist: [Name not provided] Date of Report: [Date not provided]

Notes: This report is generated based on the provided image and is intended solely for the purpose of demonstrating model capability in interpreting radiological images. This report is not a substitute for professional medical diagnosis, treatment, or professional radiological interpretation. The content of this report is not to be used for medical guidance or clinical decision-making. No identifying patient information was provided or used in the creation of this report.

[Figure 642]

###### Gemini Pro

Radiology Report Patient Name: [Patient Name] Age: [Age] Sex: [Sex] Date: [Date] Reason for Exam: Chest pain Findings: The chest X-ray shows a large mass in the upper lobe of the right lung. The mass is round and has a well-defined border. It is surrounded by a thin rim of lucency. The hilum of the lung is displaced to the right. The heart and mediastinum are unremarkable. Impressions:

[Figure 643]

SPHINX

Impression: The chest X-ray shows a normal-sized and well-defined rib cage with no evidence of any abnormalities or pathologies. The lungs are well-expanded and there are no signs of consolidation or infiltrates. The heart is well-positioned and there are no signs of cardiomegaly or cardiomegaly. Findings: The chest X-ray shows a normal-sized and well-defined rib cage with no evidence of any abnormalities or pathologies. The lungs are well-expanded and there are no signs of consolidation or infiltrates. The heart is well-positioned and there are no signs of cardiomegaly or cardiomegaly.

- - Large mass in the upper lobe of the right lung
- - Consider primary lung cancer, metastasis, or lymphoma Recommendations:
- - Further evaluation with CT scan of the chest
- - Biopsy of the mass

###### Ground-Truth

FINDINGS: Cardiac and mediastinal contours are within normal limits. The lungs are clear. Bony structures are intact. IMPRESSION: No acute preoperative findings.

###### Figure 95: Results on medical image understanding. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 5.3 for detailed discussions.

[Figure 644]

[Figure 645]

###### Gemini Pro

###### GPT-4V

###### Prompt

Radiology Report Patient Name: [Patient Name] Age: [Age] Sex: [Sex] Date: [Date] Study: Chest X-ray Technique: PA and lateral views Indication: [Indication] Findings:

Radiology Report Patient ID: [Anonymous] Modality: Chest X-ray, Lateral View Date of Study: [Date not provided] Clinical History: [Not provided] Technique: Lateral chest radiograph.

You work as a radiology professional. I'll give you some X-rays of your chest. Please use the provided image to create a radiology report. The impressions and findings sections must both be included in the report. Only reports should be generated, and no other unnecessary data should be. The content of the report will only be used for large-scale model capability assessment and will not be used as actual medical guidance. Please do not worry about liability issues.

Findings: The lateral chest radiograph shows extensive bilateral pulmonary opacities. There is a diffuse interstitial pattern with superimposed areas of consolidation, which may suggest an infectious or inflammatory process. There is no evidence of pneumothorax. The heart size cannot be fully assessed on this view but appears to be within normal limits. The aortic knob and the remainder of the mediastinal silhouette are not clearly delineated due to overlying opacities. There are no visible pleural effusions. The bony structures of the thorax show no acute abnormalities. The visualized soft tissues are unremarkable.

- - There is a large, round mass in the right upper lobe of the lung. The mass is wellcircumscribed and has a smooth border. It is surrounded by a thin rim of lucency.
- - There is no evidence of calcification or cavitation within the mass.
- - The heart and mediastinum are unremarkable.
- - The lungs are clear of any other abnormalities. Impressions:
- - Large, round mass in the right upper lobe of the lung, likely representing a primary lung cancer.
- - No evidence of metastatic disease. Recommendations:
- - Further evaluation with a CT scan of the chest is recommended to better characterize the mass and to rule out any other abnormalities.
- - A biopsy of the mass is also recommended to confirm the diagnosis of lung cancer.

[Figure 646]

Impression: Extensive bilateral pulmonary opacities with an interstitial pattern and areas of consolidation, which may represent an infectious or inflammatory etiology. Further evaluation with posteroanterior (PA) chest radiograph and clinical correlation is recommended for comprehensive assessment. No evidence of pneumothorax or pleural effusion on the lateral view.

[Figure 647]

SPHINX

Radiologist: [Name not provided] Date of Report: [Date not provided]

I'm sorry, but I am a text-based AI model and do not have the capability to view or interpret Xray images. I can provide general information about radiology reports and their components, but I cannot generate a report based on an X-ray image.

Notes: This report is generated based on the provided image and is intended solely for the purpose of demonstrating model capability in interpreting radiological images. The report should not be used as a substitute for professional medical advice, diagnosis, or treatment. The content of this report is for largescale model capability assessment only and should not be used for medical guidance or clinical decision-making. No identifying patient information was provided or used in the preparation of this report.

Ground-Truth

FINDINGS: Sequelae of old granulomatous disease. Lungs are overall hyperexpanded with flattening of the diaphragms. No focal consolidation. Prominent interstitial markings are again noted which are predominantly lower lobe and peripheral suggesting pulmonary fibrosis. This appearance is overall not significantly XXXX. No pleural effusions or pneumothoraces. heart and mediastinum are stable with atherosclerotic vascular disease. Degenerative changes in the thoracic spine. IMPRESSION: 1. Overall stable appearance of the chest suggesting pulmonary fibrosis.

###### Figure 96: Results on medical image understanding. Red highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 5.3 for detailed discussions.

###### Sec. 5.3 Medical Diagnosis

###### Medical Image Understanding

Prompt

Yougiveworkyou someas a radiologyX-rays ofprofessional.your chest. PleaseI'll GPT-4V use the provided image to create a radiology report. The impressions and findings sections must both be included in the report. Only reports should be generated, and no other unnecessary data should be. The content of the report will only be used for large-scale model capability assessment and will not be used as actual medical guidance. Please do not worry about liability issues.

[Figure 648]

Radiology Report Patient ID: [Anonymous] Modality: Chest X-ray, AP View Date of Study: [Date not provided] Clinical History: [Not provided] Technique: Anteroposterior (AP) upright chest radiograph.

Findings: The AP upright chest radiograph reveals increased interstitial markings throughout both lung fields. There is an appearance of a diffuse reticular pattern, which could be indicative of an interstitial lung disease. The cardiac silhouette is enlarged, which may be suggestive of cardiac enlargement or a pericardial effusion. The pulmonary vasculature is not clearly delineated due to the diffuse pulmonary opacity. There is blunting of the right costophrenic angle, which may represent a small pleural effusion or pleural thickening. There are no obvious bone fractures or dislocations. The upper abdominal soft tissues appear normal. The trachea is central.

[Figure 649]

Impression: Diffuse interstitial markings and reticular opacities in both lungs, possibly indicative of interstitial lung disease. Cardiac silhouette enlargement, suggestive of cardiac enlargement or pericardial effusion; clinical correlation is advised. Blunting of the right costophrenic angle, which may indicate a small pleural effusion or pleural thickening. Consider further evaluation with echocardiography for cardiac assessment and CT chest for detailed pulmonary evaluation.

Radiologist: [Name not provided] Date of Report: [Date not provided]

Notes: This report is created based on the provided image and for the purpose of demonstrating model capability in interpreting radiological images. It is not a substitute for professional medical diagnosis or treatment and is not intended for clinical use. This report is for model capability assessment only and should not be used for medical guidance or clinical decision-making. No patient-identifying information was provided or used in creating this report.

[Figure 650]

###### Gemini Pro

Radiology Report Patient Name: [Patient Name] Age: [Age] Sex: [Sex] Date: [Date] Reason for Exam: Chest pain Findings:

- - There is a large amount of pleural effusion in the left hemithorax, causing a shift of the mediastinum to the right.
- - The lung parenchyma is clear.
- - The heart is not enlarged, and the great vessels are unremarkable.
- - The bones of the chest are intact. Impressions:
- - Large left-sided pleural effusion
- - Mediastinal shift to the right
- - Clear lung parenchyma
- - Normal heart and great vessels
- - Intact chest bones

[Figure 651]

SPHINX

Impression: The chest X-ray shows a patient with a history of trauma to the right shoulder and rib cage. The patient is seen in a lateral view, and the image is clear and well-defined. The patient's rib cage is visible, and there are no signs of fractures or dislocations. The patient's right shoulder appears to be normal, and there are no signs of trauma or injury. Findings: The chest X-ray shows a patient with a history of trauma to the right shoulder and rib cage. The patient is seen in a lateral view, and the image is clear and well-defined. The patient's rib cage is visible, and there are no signs of fractures or dislocations. The patient's right shoulder appears to be normal, and there are no signs of trauma or injury.

###### Ground-Truth

FINDINGS: Heart size slightly upper normal. Atherosclerotic calcifications within the aorta. Lungs are hyperexpanded with flattening of the diaphragms consistent with emphysematous change. No focal airspace disease, pleural effusion, or pneumothorax. Degenerative changes of the spine are noted. IMPRESSION: 1. Emphysematous change without acute radiographic cardiopulmonary process.

- Figure 97: Results on medical image understanding. Green highlights the right answer. Red

highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 5.3 for detailed discussions.

###### Sec. 5.4 Economic Analysis

###### Economic Analysis

[Figure 652]

Prompt

###### Gemini Pro

This graph illustrates the historical trend of Google's stock price. On December 6th, Google released Gemini, their latest large multimodal model. What is the potential for price movement in Google's stock on December 7th?

The stock price of Google is likely to increase on December 7th due to the release of Gemini, their latest large multimodal model. This is because investors are likely to be optimistic about the potential of this new product and its ability to generate revenue for Google.

[Figure 653]

[Figure 654]

###### SPHINX

I'm sorry, but I cannot predict the future price movement of Google's stock. The stock market is influenced by various factors such as market trends, economic indicators, and company news, among others. It is important to conduct thorough research and analysis before making any investment decisions.

[Figure 655]

###### GPT-4V

The image you've uploaded shows a graph with a trend line, presumably representing Google's stock price over a period of time. However, as an AI developed by OpenAI, I don't have the ability to predict future stock prices or market movements. Stock prices can be influenced by a wide range of factors, including market sentiment, investor perceptions, and broader economic trends, not just product releases, even of significant ones like a large multimodal model.

If the release of Gemini, Google's latest large multimodal model, is perceived positively by the market and seen as a sign of innovation and potential future revenue growth, it could have a positive impact on the stock price. Conversely, if the market perceives it negatively or is indifferent, the stock price may not be significantly affected.

Investors often look for the following indicators when evaluating such events:

- 1. Market Reaction: How did the market react to the news? Was there an increase in trading volume? Did the stock price show significant movement?
- 2. Analyst Opinions: What are the financial analysts saying about the release? Do they predict increased revenue or market share as a result of the new product?
- 3. Industry Impact: Does the new model give Google a competitive advantage in the industry?
- 4. Investor Sentiment: Are investors bullish about Google's future prospects due to this release?
- 5. Press Releases and News: Is there positive media coverage about the new model?

For the specific potential of price movement on December 7th, one would need to analyze not only the historical trend but also current market conditions, news, analyst reports, and potentially the company's own forecasts or guidance provided in their release or investor relations materials.

For real-time analysis and predictions, it's best to consult a financial advisor or use real-time market analysis tools. Remember that all investments carry risk, and past performance is not indicative of future results.

- Figure 98: Results on economic analysis. Green highlights the right answer. Refer to Section 5.4 for detailed discussions.

###### Sec. 5.4 Economic Analysis

###### Economic Analysis

Prompt

On January 18, 2022, Microsoft announced its intention to acquire video game giant Activision Blizzard at a price of US$95 per share. The image is the change in Activision Blizzard's stock price after Microsoft announced its acquisition plan. Does the market expect the acquisition to be successful?

[Figure 656]

[Figure 657]

[Figure 658]

###### Gemini Pro

###### GPT-4V

Based on the provided chart, the market expects the acquisition to be successful. The stock price of Activision Blizzard increased significantly after the announcement, which indicates that investors believe the acquisition will be beneficial to the company.

The image you've provided appears to be a stock price chart for Activision Blizzard, with the intent to observe the market reaction after Microsoft announced plans to acquire the company. To determine whether the market expects the acquisition to be successful based on the stock price movement, one would look for a positive reaction in the form of a stock price increase following the announcement. This is typically because investors may expect the acquisition to provide future value.

[Figure 659]

SPHINX

I'm sorry, but I cannot answer that question as it is subjective and depends on various factors such as market sentiment, investor expectations, and the overall economic climate. The stock price of Activision Blizzard has fluctuated significantly since the announcement of the acquisition, indicating that the market is reacting to the news. However, it is

However, I cannot analyze the image directly to provide specific insights or numeric analysis. If you need a detailed analysis, you would need to provide the numeric stock price data before and after the announcement, or any specific dates or price points you're interested in. Then, I could assist with interpreting the data.

important to note that the stock market is influenced by a wide range of factors and can be unpredictable.

- Figure 99: Results on economic analysis. Green highlights the right answer. Refer to Section 5.4 for detailed discussions.

[Figure 660]

###### GPT-4V

Prompt

The image shows a construction site with an individual in the center who appears to be engaged in some activity. However, it's important to note that while I can provide a general analysis based on construction safety standards, I cannot make definitive assessments about behavior without context or motion, nor can I provide personal information about the individuals in the picture.

Please analyze the dangerous behavior of the person in the middle of the picture in this environment.

[Figure 661]

Based on the setting, here are some potential safety concerns that could be considered dangerous behavior in a construction environment:

- 1. Lack of Personal Protective Equipment (PPE): The person may not be wearing a hard hat, which is essential to protect against head injuries from falling objects or bumps against fixed objects. If they're also not wearing safety goggles, they may be at risk of eye injuries from debris.
- 2. Improper Use of Scaffolding: If the individual is manipulating or leaning on the scaffolding without proper safety measures, they could destabilize the structure or fall from a height.
- 3. Handling Materials Improperly: If the person is lifting heavy materials without proper technique or equipment, they may be at risk of musculoskeletal injuries.
- 4. Nearby Hazards: There may be tripping hazards such as loose materials or tools on the ground, and the individual could be at risk if they are not paying attention to their surroundings.
- 5. Unsecured Work Area: If the work area is not properly cordoned off, there may be a risk of unauthorized personnel entering the area and being exposed to construction hazards.

[Figure 662]

###### Gemini Pro

The person in the middle of the picture is not wearing a safety harness or a hard hat, which are

required on a construction site. He is also standing on a piece of rebar that is not attached to anything, which could cause him to trip and fall. There are also several pieces of rebar sticking out of the ground that he could trip over. The person in the middle of the picture is not wearing the proper safety gear and is not following the safety procedures that are required on a construction site.

[Figure 663]

SPHINX

The person in the middle of the picture is engaging in a dangerous behavior by working on a construction site without proper safety gear. In the image, the man is wearing gloves and a helmet, but he is not wearing any other protective equipment, such as a hard hat, safety glasses, or protective clothing. This lack of proper safety gear increases the risk of accidents and injuries while working on the construction site. It is essential for workers to follow safety guidelines and wear appropriate protective equipment to minimize the risk of harm in such environments.

It's crucial for all workers on a construction site to adhere to safety protocols, which typically include the use of PPE, proper handling of tools and materials, awareness of one's surroundings, and adherence to guidelines for working at heights or with potentially unstable structures. Regular safety training and monitoring can help mitigate these risks.

- Figure 100: Results on surveillance and security. Green highlights the right answer. Red highlights

the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 5.5 for detailed discussions.

Prompt

Please analyze the dangerous behaviors of the two individuals in the environment shown in the picture.

[Figure 664]

[Figure 665]

###### GPT-4V

The image shows a construction site with two individuals engaged in potentially unsafe behaviors.

The person standing appears to be using a hammer or similar tool without wearing any visible personal protective equipment (PPE) such as gloves, safety glasses, or a hard hat. The absence of such equipment poses a risk of injury from flying debris, accidental strikes, or falls.

[Figure 666]

###### Gemini Pro

The individual crouched down is also not wearing a hard hat, which is a critical piece of PPE on a construction site to protect against head injuries from falling objects or bumps against hard surfaces. Additionally, this person's position appears unbalanced, increasing the risk of a fall, and they seem to be working close to unguarded edges, which could lead to a serious fall if one were to occur.

The two individuals are not wearing the appropriate safety gear. They are both wearing sandals, which are not safe for a construction site. The man on the left is not wearing a hard hat, which is also a safety hazard. The man on the right is holding a measuring tape, which could be a tripping hazard. They are also both standing in a dangerous area, as there are exposed rebar rods and other construction materials that could cause injury.

Both individuals are also wearing casual clothing, which might not provide adequate protection against the hazards typical of a construction environment, such as sharp objects or concrete burns.

[Figure 667]

SPHINX

In the image, there are two individuals working on a construction site. One of the workers is holding a cell phone, which is a dangerous behavior in a construction environment. Using a cell phone while working on a construction site can lead to distractions, decreased productivity, and increased risk of accidents. It is essential for workers to remain focused and attentive to their surroundings to ensure their safety and the safety of others around them. The second worker is holding a hammer, which is a necessary tool for construction work. However, it is important to handle the hammer with care and follow proper safety procedures to avoid injuries.

Lastly, the construction site itself appears to have some safety issues, such as exposed rebar and open edges, which should be properly marked and covered to prevent accidents.

It is crucial for safety protocols to be followed meticulously to ensure the well-being of all individuals on the site. This includes proper use of PPE, ensuring the work environment is safe and hazards are mitigated, and that workers are trained to work safely and are aware of their surroundings at all times.

- Figure 101: Results on surveillance and security. Green highlights the right answer. Red highlights the wrong answer. Refer to Section 5.5 for detailed discussions.

###### Sec. 5.6 Remote Sensing Image Analysis

###### Remote Sensing Image Analysis

[Figure 668]

###### Gemini Pro

Prompt

The scene is an aerial view of an airport. There are four airplanes on the runway, and one airplane is parked at the gate. The airplanes are all white with different airline logos. The runway is surrounded by taxiways and buildings.

Please describe this scene in details.

[Figure 669]

[Figure 670]

GPT-4V

The image is an aerial view of a section of an airport. There are two large commercial airplanes

on a tarmac area, positioned as if they are parked or preparing for taxi. They are near what appears to be a terminal building with jet bridges, indicating passenger boarding areas. The airplanes are white, with the tail sections visible and pointed towards the terminal building. There are various markings on the ground, likely designating taxiways and parking spots for the aircraft. The tarmac shows some signs of wear and tire marks. On the left side of the image, there is a runway with clear runway markings. The area is devoid of any significant vegetation, and there are no visible people, suggesting the photo was taken from a considerable height. Shadows cast by the airplanes suggest a light source from the top left of the image, indicating the time of day could be either morning or afternoon. The overall scene is one that is typical for an operational area within a large airport, dedicated to aircraft movements and passenger boarding.

[Figure 671]

SPHINX

The image is an aerial view of a large airport, showcasing several planes parked on the tarmac. There are a total of four planes in the scene, with two planes positioned on the left side, one in the middle, and another on the right side of the airport. The planes are of various sizes and orientations, indicating a busy and active airport environment.

Figure 102: Results on remote sensing image analysis. Green highlights the right answer. Red highlights the wrong answer. Yellow highlights the incompetence in performing the task. Refer to Section 5.6 for detailed discussions.

###### Sec. 5.6 Remote Sensing Image Analysis

###### Remote Sensing Image Analysis

[Figure 672]

###### GPT-4V

Prompt

The image you've provided appears to be an aerial or satellite view of a landscape that includes both urban and rural areas. Here's a detailed analysis of different elements within the image:

Please analyze everything in this image in details.

[Figure 673]

- 1. Urban Areas: These are characterized by the dense clustering of buildings and roads. The urban sections display a variety of building sizes and arrangements, indicating a mix of residential, commercial, and possibly industrial zones. There's a grid-like pattern in some parts of the urban areas, which suggests planned urban development.
- 2. Rural Areas: The green patches represent agricultural fields, indicating a rural landscape. The fields have different colors, ranging from light to dark green, which might be due to different types of crops or stages of growth.
- 3. Water Bodies: The dark, snaking lines seem to be rivers or canals. These water bodies vary in width and flow through both the urban and rural areas, which could be significant for irrigation, transportation, or as natural boundaries.
- 4. Road Networks: The grey lines traversing both the rural and urban areas are likely roads or highways. Some of these roads connect the rural areas to the urban centers, facilitating the movement of goods and people.
- 5. Vegetation: Apart from the agricultural fields, there are areas with darker green, likely representing trees or forests. These might be natural or planted forests and can serve as windbreaks, recreational areas, or for conservation purposes.
- 6. Patterns of Settlement: The arrangement of the buildings in the urban areas varies, with some sections showing a more organized layout while others are more haphazard. This could indicate different phases of development or variations in urban planning.
- 7. Topography: Although it's challenging to determine elevation from this image, the way the land is used and the course of the water bodies might suggest flat to gently rolling terrain.
- 8. Infrastructure: There are some large structures that could be facilities like factories, warehouses, or large commercial setups, especially at the outskirts of the urban areas.
- 9. Shadows: The image does not show visible shadows that would help determine the time of day or the direction of the sun, which is typical in high-resolution satellite imagery where the sun angle is high to minimize shadow.

[Figure 674]

###### Gemini Pro

The image is a satellite image of an area in Japan. The image is in false color, with

vegetation appearing in bright green and water appearing in dark blue. The image is oriented with north at the top. The image shows a large area of agricultural land, with many small fields of different crops. There are also several small towns and villages in the image, as well as a large river that runs through the center of the image. The image is taken from a high resolution satellite, and the details of the landscape are clearly visible.

The image does not provide any specific geographic landmarks that would allow for the identification of the exact location. It gives a comprehensive view of a region that includes both densely populated areas and agricultural land, indicating a region that supports both urban living and farming activities.

[Figure 675]

SPHINX

The image is a valuable resource for studying the geography of Japan. It can be used to identify different types of land use, as well as to study the impact of human activity on the environment. The image can also be used to create maps and other visual representations of the area.

The image is a large green and gray map of a city. The map is divided into several sections, each with its own unique features. The city is surrounded by green fields and a river, which runs through the middle of the map. The river is visible in the upper and lower parts of the map, and it appears to be a significant feature of the landscape. The map also shows a bridge, which is located near the top left corner of the image. The city is densely populated, with many buildings and structures visible throughout the map.

Figure 103: Results on remote sensing image analysis. Green highlights the right answer. Refer to Section 5.6 for detailed discussions.

[Figure 676]

This is the component diagram of the mobile phone. Please tell me GPT-4V step by step how to assemble it into a mobile phone.

Assembling a smartphone from its component parts is a complex process that typically requires specialized tools and knowledge. However, I can provide you with a simplified overview of the steps involved in assembling a smartphone from its components:

[Figure 677]

- 1. Prepare Your Workspace: Ensure you have a clean, organized, and static-free workspace. Use a mat that's designed for electronic repairs.
- 2. Battery Preparation: Begin with the battery component. If the battery is already installed in the frame, leave it as is; if not, it would typically be one of the last components you install to avoid any risk during the assembly.
- 3. Display and Digitizer: The display is usually pre-assembled with the digitizer (the touch-sensitive part). This is carefully positioned and connected to the main board via a ribbon cable.
- 4. Main Board: The main board, or motherboard, is the central component. Carefully place it in the frame, ensuring no cables are trapped underneath. Connect any ribbon cables from the display and digitizer to the main board.
- 5. Front-Facing Camera: Install the front-facing camera in the designated spot on the frame or main board. Connect it with the main board via its ribbon cable.
- 6. Rear Camera: The rear camera is installed similarly to the frontfacing camera, secured in place, and connected to the main board.
- 7. Speakers and Microphones: Install the speaker modules and microphones in their respective places, making sure they're properly aligned with the frame for optimal sound quality.
- 8. SIM Card Tray: Insert the SIM card tray in its slot. This is usually on the side of the frame.
- 9. Buttons and Switches: The physical buttons and switches are placed in their respective spots and connected to the main board, usually with very small screws or clips.
- 10. Additional Sensors and Antennas: Install any additional components like proximity sensors, ambient light sensors, and antennas, ensuring they are properly connected to the main board.
- 11. Battery Connection: If not already done, now is the time to install the battery. Connect it to the main board with its connector.
- 12. Final Assembly: Carefully close the phone by attaching the back cover. This might involve carefully placing it and securing it with screws or adhesive, depending on the model.
- 13. Testing: Once everything is assembled, it's important to test all the functions of the phone to ensure everything is working properly.

[Figure 678]

###### Gemini Pro

- 1. Prepare the parts. Before you begin, make sure you have all of the necessary parts. This includes the phone's body, the display, the battery, the motherboard, the camera, and the speakers.
- 2. Install the motherboard. The motherboard is the main circuit board of the phone. It connects all of the different parts of the phone together. To install the motherboard, first, remove the back cover of the phone. Then, locate the motherboard and carefully remove it from the phone. Be careful not to damage the motherboard.
- 3. Install the battery. The battery is the power source for the phone. To install the battery, first, remove the back cover of the phone. Then, locate the battery and carefully remove it from the phone. Be careful not to damage the battery.
- 4. Install the display. The display is the screen of the phone. To install the display, first, remove the back cover of the phone. Then, locate the display and carefully remove it from the phone. Be careful not to damage the display.
- 5. Install the camera. The camera is the device that takes pictures and videos. To install the camera, first, remove the back cover of the phone. Then, locate the camera and carefully remove it from the phone. Be careful not to damage the camera.
- 6. Install the speakers. The speakers are the devices that emit sound. To install the speakers, first, remove the back cover of the phone. Then, locate the speakers and carefully remove them from the phone. Be careful not to damage the speakers.
- 7. Reassemble the phone. Once you have installed all of the parts, you can reassemble the phone. To do this, first, replace the back cover of the phone. Then, carefully put the phone back together. Be careful not to damage the phone.
- 8. Test the phone. Once you have reassembled the phone, you can test it to make sure that it works properly. To do this, turn on the phone and check the display, the camera, the speakers, and the battery. Make sure that everything is working properly.

Keep in mind that this is a very high-level overview. Each smartphone can have a different assembly process due to variations in design. Professional assembly would also involve using specific tools to handle the components without damaging them, applying thermal paste or pads as needed for heat management, and ensuring all the connectors are properly seated. If you're attempting to assemble a smartphone yourself, it's highly recommended to follow a guide specific to your phone model and use the appropriate tools for the job.

[Figure 679]

SPHINX

I'm sorry, but I cannot provide a step-by-step guide on assembling a mobile phone from its components as it requires specialized knowledge and tools. Additionally, taking apart a mobile phone can void its warranty and potentially damage the device. It is recommended to seek professional assistance or follow the manufacturer's instructions for repairing or replacing components in a mobile phone.

Figure 104: Results on robot motion planning. Red highlights the wrong answer. Refer to Section 5.7 for detailed discussions.

The picture shows the parts of a chair, please tell me step by step how to assemble these parts into a chair, and please describe in detail the shape and function of the chair after assembling.

[Figure 680]

[Figure 681]

Gemini Pro

- 1. Assemble the chair base. The chair base is the part of the chair that supports the seat and backrest. To assemble the chair base, first insert the casters into the holes in the bottom of the base. Next, attach the gas lift cylinder to the center of the base. Finally, screw the armrests into the holes on the sides of the base.
- 2. Assemble the seat. The seat of the chair is the part that you sit on. To assemble the seat, first attach the seat cushion to the seat frame. Next, screw the seat frame to the gas lift cylinder. Finally, attach the armrests to the sides of the seat.
- 3. Assemble the backrest. The backrest of the chair is the part that supports your back. To assemble the backrest, first attach the backrest cushion to the backrest frame. Next, screw the backrest frame to the seat frame. Finally, attach the headrest to the top of the backrest.
- 4. Adjust the chair. Once the chair is assembled, you can adjust it to fit your needs. You can adjust the height of the seat by raising or lowering the gas lift cylinder. You can also adjust the tilt of the backrest by turning the tilt knob.

[Figure 682]

GPT-4V

To assemble the chair from the parts shown in the image, follow these steps:

###### 1. Base Assembly:

- Start with the star-shaped base, which is typically the bottom of the chair. Attach the casters (wheels) into each of the five sockets on the star base. They usually click into place.

###### 2. Gas Lift and Seat Attachment:

- - Insert the gas lift cylinder (the metallic rod) into the center of the star base. This part allows the chair to adjust in height.
- - Slide the smaller cylindrical sleeve (if present) over the gas lift for aesthetic purposes. 3. Seat Mechanism:
- - Position the seat mechanism (the part with levers and knobs) onto the bottom of the seat. The mechanism allows you to adjust the tilt and lock the chair in place.
- - Secure the mechanism to the seat using the bolts or screws provided.

The chair is a comfortable and stylish piece of furniture that is perfect for any home or office. The chair is made of high-quality materials and is built to last. The chair is also easy to assemble and adjust.

[Figure 683]

SPHINX

I can provide you with a general overview of the chair assembly process and the chair's features.

###### 4. Connecting Seat to Base:

- Once the mechanism is secured to the seat, place the seat onto the top of the gas lift cylinder. It should fit onto the cylinder and allow the seat to move up and down.

First, take the backrest and seat cushion and place them on the chair frame. Next, attach the armrests to the chair frame. Install the base of the chair by connecting the star-shaped leg to the chair frame. Attach the chair's wheels to the base of the chair. Finally, connect the chair's handle to the chair frame. the chair after assembling has the following features:

###### 5. Backrest Attachment:

- Attach the backrest to the seat. There may be a bracket or screws that will help you secure the backrest to the seat or directly to the mechanism.

###### 6. Armrests and Additional Features:

- - If there are armrests, they would typically screw into the underside of the seat or sides of the backrest.
- - Add any additional features like headrests or lumbar support cushions as per the design of the chair. Once assembled, the chair should be a comfortable office chair with an ergonomic design, adjustable height, and tilt functions. It will have a padded seat and backrest for support, a five-point wheeled base for easy movement, and possibly adjustable armrests for additional comfort. The materials appear to be a combination of metal for the base and structure, with mesh and padded upholstery for breathability and comfort. The chair is designed to provide proper support during long periods of sitting, making it ideal for office work.

The chair is designed with a star-shaped leg, which provides stability and support. The chair has wheels, which allow for easy movement and mobility. The chair has an adjustable handle, which allows for customization of the chair's height and comfort. The chair has a backrest and seat cushion, which provide support and comfort for the user. The chair has armrests, which provide additional support and comfort for the user. overall, the chair is designed to provide comfort and support for the user while they sit and work.

- Figure 105: Results on robot motion planning. Yellow highlights the incompetence in performing the task. Refer to Section 5.7 for detailed discussions.

###### 6 Quantitative Experiments

In the preceding sections, we have demonstrated the qualitative performance of Gemini, GPT-4V, and Sphinx on a selection of representative samples to intuitively illustrate their visual expertise. Building upon this foundation, this section delves into the quantitative analysis of the models’ capabilities. Specifically, we report comprehensive results obtained from the popular MME benchmark [19], offering an empirical evaluation of their performance in various scenarios.

###### 6.1 MME Benchmark

Instruction design. The instruction design of MME [19] is uniquely structured to elicit straightforward ‘yes’ or ‘no’ responses. Each test image is accompanied by two carefully crafted instructions. These instructions are distinguished primarily by their questions – one is designed with a ‘yes’ as the correct answer, and the other with a ‘no’. This binary questioning format is pivotal in assessing the MLLM’s comprehension abilities. A model that correctly answers both questions demonstrates not only a clear understanding of the visual content but also an underlying grasp of the associated contextual knowledge. This approach ensures a more robust evaluation of the MLLM’s interpretative capabilities in relation to visual stimuli.

Evaluation metric. The assessment framework is tailored to align with the model’s binary output options: ‘yes’ or ‘no’. This dichotomy allows for a straightforward computation of two key metrics: standard accuracy and enhanced accuracy (accuracy+). Standard accuracy is determined question-byquestion, whereas accuracy+ is a more rigorous measure, requiring correct answers to both questions associated with a single image. Notably, the baseline random accuracies for these metrics are 50% and 25%, respectively, highlighting the heightened precision demanded by accuracy+. This dual-metric approach not only quantifies the model’s basic accuracy but also provides a more nuanced insight into its comprehensive understanding of the visual content. Further, the evaluation involves aggregating these metrics; the score for each sub-task is the sum of both accuracy measures. Similarly, the overall perception and cognition scores are derived from the cumulative scores of their respective sub-tasks, offering a multi-faceted evaluation of the model’s performance.

Data collection. For the perceptual tasks of Existence, Count, Position, Color, OCR, Poster, Celebrity, Scene, Landmark, and Artwork, the sample images are sourced from publicly available datasets [34, 28, 40, 64, 55]. Regarding the cognitive tasks of Commonsense Reasoning, Numerical Calculation, Text Translation, and Code Reasoning, the sample images are obtained from manually photographed or diffusion model-generated sources.

###### 6.2 Results

As shown in Table 1, in terms of the comprehensive performance of perception and cognition, Gemini exhibits superior performance with a score of 1933.4, closely followed by the GPT-4V model, which scored 1926.6. The Sphinx model trails with a score of 1870.2.

Perception. Sphinx surpasses other models in most of the perception tasks. This is particularly evident in the task of position perception, where Gemini and GPT-4V underperform Sphinx by a margin of 60 points. This observation diverges somewhat from the rankings illustrated by the qualitative experiments in Section 2. We hypothesize that this discrepancy arises because the samples used in the MME to evaluate perception primarily originate from public academic datasets [34, 28, 40, 64, 55], whose data distribution closely aligns with the training set of the open-source model Sphinx. Furthermore, it is noteworthy that due to the refusal to provide information related to real persons, the score of GPT-4V in the sub-task of celebrity recognition is zero.

Cognition. GPT-4V dominates almost all cognition tasks, especially code reasoning, with a notably high score of 170.0. This finding is largely consistent with the comparative results of the qualitative experiments discussed in Section 3.

In summary, while each model demonstrates particular strengths across various tasks within the benchmark, Gemini and GPT-4V outperform Sphinx when considering the overall performance. Notably, GPT-4V exhibits leading performance in cognition tasks, while Gemini demonstrates a more balanced performance across various tasks, thereby achieving the highest score. This aspect is intuitively illustrated in Figure 106.

[Figure 684]

- Figure 106: Evaluation on 14 sub-tasks of the MME benchmark. We observe the following phenomena: (1) GPT-4V refuses to respond to names about real people, leading to a zero score in the celebrity recognition sub-task. (2) Both Gemini and GPT-4V exhibit suboptimal performance in the position recognition sub-task, aligning with the qualitative findings from earlier experiments shown in Figure 1, suggesting that the two models may be insensitive to spatial information. (3) The performance of Sphinx on perception is on par or even exceeds that of Gemini and GPT-4V. This is probably because that Sphinx pay more attention on perception during training, such as object detection. In contrast, compared to Gemini and GPT-4V, Sphinx lags considerably on the cognition sub-tasks, including commonsense reasoning, numerical calculation, text translation, and code reasoning.

Perception Cognition

Model Overall

Exist. Count Pos. Color Poster Cele. Scene Land. Art. OCR Com. Cal. Trans. Code

Sphinx [35] 1870.2 195.0 160.0 153.3 160.0 164.3 177.9 160.0 168.1 134.0 87.5 130.0 55.0 75.0 50.0 GPT-4V [43] 1926.6 190.0 160.0 95.0 150.0 192.2 0.0 151.0 138.3 148.0 185.0 142.1 130.0 75.0 170.0 Gemini [21] 1933.4 175.0 131.7 90.0 163.3 165.0 147.4 144.8 158.8 135.8 185.0 129.3 77.5 145.0 85.0

Table 1: Evaluation on the MME benchmark. Here we report the results on all the sub-tasks, including Existence (Exist.), Count, Position (Pos.), Color, OCR, Poster, Celebrity (Cele.), Scene, Landmark (Land.), Artwork (Art.), Commonsense Reasoning (Com.), Numerical Calculation (Cal.), Text Translation (Trans.), and Code Reasoning (Code). The highest scores across individual sub-tasks are highlighted in bold.

###### 7 Conclusion

###### 7.1 Summary

In this report, we have conducted a comprehensive evaluation of three powerful MLLMs, i.e., Gemini Pro [21], GPT-4V [43], and Sphinx [35], which involves diverse qualitative samples and a quantitative benchmark, MME [19]. For multi-faceted comparison, we carefully collect numerous samples covering different domains of visual understanding, including fundamental perception, advanced cognition, challenging vision tasks, and various expert capacities. Each domain also contains several subtasks for in-depth discussion and analysis.

###### 7.2 Gemini vs GPT-4V

The qualitative results indicate that Gemini is indeed a strong challenger to GPT-4V, given its superior multi-modal reasoning capacity. In most cases, Gemini achieves competitive answering accuracy compared to GPT-4V, and showcases different response styles and preferences.

Differences. For comparison, GPT-4V tends to generate more detailed descriptions of the perception tasks (Figures 8, 9, 10, 23, 25), and provide in-depth analysis with step-by-step intermediate reasoning for the cognition tasks (Figures 39, 42, 48, 65, 69). Instead, Gemini prefers to provide a direct and concise response to the answer, which helps the users to rapidly locate pertinent information. When there are a greater number of visual elements in the image (Figures 8, 9), the fine-grained perception advantages of GPT-4V become more pronounced, which provides more accurate recognition of visual details. However, GPT-4V, due to privacy concerns, may decline to respond to topics related to celebrities (Figures 24, 29, 81 and the Celebrity metric on the MME benchmark [19] shown in Figure 106), or it may refrain from attempting to answer certain out-of-scope questions by anticipating its own limitations (Figures 31, 77). For some specific vision and expert-level tasks (Figures 37, 38, 85), Gemini normally demonstrates a broader range of learned knowledge and generalization capabilities, indicating better applicability across various domains.

Common issues. There are four common issues of the two MLLMs. 1) The first limitation is the spatial perception capabilities. From the qualitative perception examples (Figure 1) and quantitative results on the MME benchmark (the Position metric shown in Figure 106), both Gemini and GPT4V are not proficient in determining the relative positions of objects. 2) The second issue is the unsatisfactory OCR (Figures 41, 45) and abstract visual understanding (Figures 50, 52). For example, they may misinterpret some of the numbers and characters in the diagrams or charts, and have difficulty comprehending some geometric shapes and abstract inductive abilities. 3) The third inadequacy lies in the logical self-consistency within reasoning. For some scientific problems (Figure 62) or ‘Yes or No’ questions (Figure 43), they occasionally provide intermediate reasoning steps that do not align with or are contrary to the final answer. 4) The fourth common issue concerns their robustness to prompt designs. As shown in Figures 43 and 59, for different approaches to framing the same question prompt, GPT-4V and Gemini would be disturbed to generate opposite answers. Such a issue affects the output stability, and impedes their further applications. We can see that both Gemini and GPT-4V still struggle in many cases, showing the long road to the general MLLM.

###### 7.3 Gemini vs Sphinx

Despite that Sphinx is on par with GPT-4V and Gemini in some cases, it is not capable of generating as consistent high-quality answers as them. This demonstrates that the open-source MLLMs still have some non-negligible gaps to closed-source models.

Failure cases of Sphinx. We observe that the failure cases of Sphinx are mainly due to two reasons. 1) The first is that the diversity of Sphinx’s training data is still lacking in some domains, constraining its capability for a wider range of tasks, e.g., scientific knowledge perception (Figure 16), visual code generation of HTML codes (Figure 46), and abstract visual reasoning (Figure 50). This motivates us to further incorporate more data in diverse domains for training faithful open-source MLLMs. 2) The second limitation is the inherent reasoning upper bound of the underlying LLM. Sphinx adopts the vanilla LLaMA-2-7B [50] for initialization, which falls short compared to larger-scale LLMs (e.g., 70B models) in certain complex tasks.

###### 7.4 Future Directions

With our comprehensive comparison and discussion, Gemini and GPT-4V are both pioneers of MLLMs in this era, showcasing sparks of artificial general intelligence [59, 9]. Looking ahead, future development of MLLMs can focus on three aspects: visual representation encoding (fine-grained appearances, spatial relation awareness), multi-modal alignment (alleviating hallucination, OCR accuracy), and LLMs’ reasoning capacity (quantitative processing, logical self-consistency). Overall, despite the exciting achievements made thus far, there still remains a considerable distance towards artificial general intelligence. We also anticipate the emergence of stronger and more comprehensive MLLMs in the future.

###### References

- [1] Chatgpt can now see, hear, and speak. https://openai.com/blog/ chatgpt-can-now-see-hear-and-speak, 2023.
- [2] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. NeurIPS, 2022.
- [3] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.
- [4] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luˇci´c, and Cordelia Schmid. Vivit: A video vision transformer. In ICCV, 2021.
- [5] Anas Awadalla, Irena Gao, Joshua Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo, March 2023.
- [6] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.
- [7] Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. Mvtec ad–a comprehensive real-world dataset for unsupervised anomaly detection. In CVPR, 2019.
- [8] Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In NeurIPS, 2020.
- [9] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023.
- [10] Holger Caesar, Varun Bankiti, Alex H Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In CVPR, 2020.
- [11] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In ECCV, 2020.
- [12] Wei-Ge Chen, Irina Spiridonova, Jianwei Yang, Jianfeng Gao, and Chunyuan Li. Llavainteractive: An all-in-one demo for image chat, segmentation, generation and editing. arXiv preprint arXiv:2311.00571, 2023.
- [13] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.
- [14] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023.
- [15] Dina Demner-Fushman, Marc D Kohli, Marc B Rosenman, Sonya E Shooshan, Laritza Rodriguez, Sameer Antani, George R Thoma, and Clement J McDonald. Preparing a collection of radiology examinations for distribution and retrieval. Journal of the American Medical Informatics Association, 2016.
- [16] Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence. Palm-e: An embodied multimodal language model. In arXiv preprint arXiv:2303.03378, 2023.

- [17] Christoph Feichtenhofer. X3d: Expanding architectures for efficient video recognition. In CVPR, 2020.
- [18] Christoph Feichtenhofer, Axel Pinz, and Andrew Zisserman. Convolutional two-stream network fusion for video action recognition. In CVPR, 2016.
- [19] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.
- [20] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010, 2023.
- [21] Google Gemini Team. Gemini: A family of highly capable multimodal models. 2023.
- [22] Ross Girshick. Fast r-cnn. In ICCV, 2015.
- [23] Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. Multimodal-gpt: A vision and language model for dialogue with humans, 2023.
- [24] Ziyu Guo, Renrui Zhang, Xiangyang Zhu, Yiwen Tang, Xianzheng Ma, Jiaming Han, Kexin Chen, Peng Gao, Xianzhi Li, Hongsheng Li, et al. Point-bind & point-llm: Aligning point cloud with multi-modality for 3d understanding, generation, and instruction following. arXiv preprint arXiv:2309.00615, 2023.
- [25] Jiaming Han, Renrui Zhang, Wenqi Shao, Peng Gao, Peng Xu, Han Xiao, Kaipeng Zhang, Chris Liu, Song Wen, Ziyu Guo, et al. Imagebind-llm: Multi-modality instruction tuning. arXiv preprint arXiv:2309.03905, 2023.
- [26] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.
- [27] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.
- [28] Qingqiu Huang, Yu Xiong, Anyi Rao, Jiaze Wang, and Dahua Lin. Movienet: A holistic dataset for movie understanding. In ECCV, 2020.
- [29] Qiuyuan Huang, Zhe Gan, Asli Celikyilmaz, Dapeng Wu, Jianfeng Wang, and Xiaodong He. Hierarchically structured reinforcement learning for topically coherent visual story generation. In AAAI, 2019.
- [30] Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al. Language is not all you need: Aligning perception with language models. arXiv preprint arXiv:2302.14045, 2023.
- [31] Huaizu Jiang and Erik Learned-Miller. Face detection with the faster r-cnn. In FG, 2017.
- [32] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. NeurIPS, 2022.
- [33] Yan Li, Bin Ji, Xintian Shi, Jianguo Zhang, Bin Kang, and Limin Wang. Tea: Temporal excitation and aggregation for action recognition. In CVPR, 2020.
- [34] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECCV, 2014.
- [35] Ziyi Lin, Chris Liu, Renrui Zhang, Peng Gao, Longtian Qiu, Han Xiao, Han Qiu, Chen Lin, Wenqi Shao, Keqin Chen, et al. Sphinx: The joint mixing of weights, tasks, and visual embeddings for multi-modal large language models. arXiv preprint arXiv:2311.07575, 2023.

- [36] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.
- [37] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023.
- [38] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating math reasoning in visual contexts with gpt-4v, bard, and other large multimodal models. arXiv e-prints, pages arXiv–2310, 2023.
- [39] Wenhan Luo, Junliang Xing, Anton Milan, Xiaoqin Zhang, Wei Liu, and Tae-Kyun Kim. Multiple object tracking: A literature review. Artificial intelligence, 293:103448, 2021.
- [40] Hui Mao, Ming Cheung, and James She. Deepart: Learning joint representations of visual arts. In ACM MM, 2017.
- [41] Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. Cross-task generalization via natural language crowdsourcing instructions. In ACL, 2022.
- [42] OpenAI. Gpt-4 technical report, 2023.
- [43] OpenAI. Gpt-4v(ision) system card. 2023.
- [44] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. NeurIPS, 2022.
- [45] Bryan A Plummer, Arun Mallya, Christopher M Cervantes, Julia Hockenmaier, and Svetlana Lazebnik. Phrase localization and visual relationship detection with comprehensive imagelanguage cues. In ICCV, 2017.
- [46] Tanzila Rahman, Hsin-Ying Lee, Jian Ren, Sergey Tulyakov, Shweta Mahajan, and Leonid Sigal. Make-a-story: Visual memory conditioned consistent story generation. In CVPR, 2023.
- [47] Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. You only look once: Unified, real-time object detection. In CVPR, 2016.
- [48] Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, et al. Multitask prompted training enables zero-shot task generalization. In ICLR, 2021.
- [49] Linsen Song, Wayne Wu, Chaoyou Fu, Chen Qian, Chen Change Loy, and Ran He. Pareidolia face reenactment. In CVPR, 2021.
- [50] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [51] Maria Tsimpoukelli, Jacob L Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. Multimodal few-shot learning with frozen language models. NeurIPS, 2021.
- [52] Qiang Wang, Li Zhang, Luca Bertinetto, Weiming Hu, and Philip HS Torr. Fast online object tracking and segmentation: A unifying approach. In CVPR, 2019.
- [53] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In ICLR, 2022.
- [54] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 2022.

- [55] Tobias Weyand, Andre Araujo, Bingyi Cao, and Jack Sim. Google landmarks dataset v2-a large-scale benchmark for instance-level recognition and retrieval. In CVPR, 2020.
- [56] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519, 2023.
- [57] Yi Wu, Jongwoo Lim, and Ming-Hsuan Yang. Online object tracking: A benchmark. In CVPR, 2013.
- [58] Shuo Yang, Ping Luo, Chen-Change Loy, and Xiaoou Tang. Wider face: A face detection benchmark. In CVPR, 2016.
- [59] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421, 2023.
- [60] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.
- [61] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A survey on multimodal large language models. arXiv preprint arXiv:2306.13549, 2023.
- [62] Licheng Yu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin Lu, Mohit Bansal, and Tamara L Berg. Mattnet: Modular attention network for referring expression comprehension. In CVPR, 2018.
- [63] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023.
- [64] Bolei Zhou, Agata Lapedriza, Jianxiong Xiao, Antonio Torralba, and Aude Oliva. Learning deep features for scene recognition using places database. NeurIPS, 2014.
- [65] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

