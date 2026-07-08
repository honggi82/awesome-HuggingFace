## TinyLVLM-eHub: Towards Comprehensive and Efficient Evaluation for Large Vision-Language Models

Wenqi Shao∗1, Meng Lei∗1,4, Yutao Hu∗1,2, Peng Gao∗1, Peng Xu1,2, Kaipeng Zhang1, Fanqing Meng1, Siyuan Huang1, Hongsheng Li1,3, Yu Qiao1 , Senior Member, IEEE, Ping Luo2,1

### arXiv:2308.03729v2[cs.CV]10Aug2024

Abstract—Large Vision-Language Models (LVLMs) have made significant strides in various multimodal tasks. Notably, GPT4V, Claude, Gemini, and others showcase exceptional multimodal capabilities, marked by profound comprehension and reasoning skills. This study introduces a comprehensive and efficient evaluation framework, TinyLVLM-eHub, to assess LVLMs’ performance, including proprietary models. TinyLVLM-eHub covers six key multimodal capabilities, such as visual perception, knowledge acquisition, reasoning, commonsense understanding, object hallucination, and embodied intelligence. The benchmark, utilizing 2.1K image-text pairs, provides a user-friendly and accessible platform for LVLM evaluation. The evaluation employs the ChatGPT Ensemble Evaluation (CEE) method, which improves alignment with human evaluation compared to word-matching approaches. Results reveal that closed-source API models like GPT4V and GeminiPro-V excel in most capabilities compared to previous open-source LVLMs, though they show some vulnerability in object hallucination. This evaluation underscores areas for LVLM improvement in real-world applications and serves as a foundational assessment for future multimodal advancements. Find our project at https://github.com/OpenGVLab/Multi-Modality-Arena.

Index Terms—Large Vision-Language Models, Multimodal Evaluation Benchmark, Evaluation Method

I. INTRODUCTION

# L

ARGE Vision-Language Models (LVLMs) have demonstrated remarkable success in various multimodal appli-

cations, including visual complex reasoning [1], [2], visual conversation [3], [4], and medical visual question answering [5], [6]. The proliferation of various LVLMs has significantly advanced our comprehension and pushed the boundaries of multimodal applications [7] across various domains [6].

LVLM is typically constructed by incorporating a Large Language Model (LLM) with a pre-trained visual encoder which facilitates the integration of images as input data [8], [9]. For example, LLaVA [3], LLaMA-Adapter V2 [4], and Otter [2] feed LLM such as Vicuna [10] or LLaMA [11] with visual tokens extracted by visual encoder ViTL/14 [12]. Notably, closed-source models such as GPT4V [13] and GeminiPro-V [14], distinguish themselves with their

1 OpenGVLab, Shanghai AI Laboratory, Shanghai, China 2Department of Computer Science, The University of Hong Kong, Hong Kong, China, 3Department of Electronic Engineering, The Chinese University of Hong Kong, Hong Kong, China 4Academy for Advanced Interdisciplinary Studies, Peking University, Beijing, China

∗ Equal First Authors

Corresponding Authors: qiaoyu@pjlab.org.cn; pluo@cs.hku.hk Project Page: http://lvlm-ehub.opengvlab.com/

[Figure 1]

Fig. 1: The overall multimodal score of 16 LVLMs in TinyLVLM-eHub obtained by averaging over six capabilities. The capability scores of top-performing LVLMs are visualized. We see that closed-source API models such as GPT4V and GeminiPro-V perform well in our benchmark. However, they face more severe hallucination issues than the open-source InstrucBLIP model.

exceptional multimodal capabilities. Although detailed model configurations are unknown, the open-source APIs of these proprietary models have demonstrated the multimodal ability. This groundbreaking development marks a significant step forward in the field of artificial general intelligence (AGI).

Despite the great success, it is vital to understand LVLMs’ capabilities in various multimodal tasks. Recent work [15],

- [16] attributes the success of LVLM to the representational power of the visual encoder, proper alignment between vision and language [1], [8], and visual instructional tuning of LLM
- [17]. However, a comprehensive evaluation of LVLMs remains underdeveloped. Another line of research [18]–[20] investigates various multimodal capabilities by experimenting with a large number of text-related visual benchmarks. Nevertheless, these studies could not assess LVLMs’ abilities in the open-set setting because they constrain the model’s answer to be close-set options such as Yes/No and A/B/C/D. Moreover, they often comprise amounts of test samples while assessing a limited number of multimodal capabilities as shown in Table I, making it incomplete and cumbersome for evaluating LVLMs.

In this work, we propose TinyLVLM-eHub to systematically evaluate various multimodal capabilities of numerous Large Vision-Language Models (LVLMs). Towards this goal, we consolidate 42 standard text-related visual benchmarks, from

each of which 50 examples are sampled. TinyLVLM-eHub is a lightweight version of LVLM-eHub [16] but enjoys several intrinsic properties. First, despite the simplicity, it can assess six categories of multimodal capabilities of LVLMs by inheriting advantages from LVLM-eHub as verified by Table VIII. Second, TinyLVLM-eHub employs ChatGPT as a judge to assess the model’s prediction and reference answer in an open-set scenario. By creating a prompt template with critical thinking, the resulting judgment aligns better with Human evaluation than the word-matching approach adopted by prior studies [16], [21] (see Table IX). Third, LVLM-eHub has only 2.1K image-text pairs. Thus it is convenient for practitioners to evaluate their offline LVLMs on TinyLVLM-eHub.

The comprehensive evaluation of TinyLVLM-eHub shows (see Fig. 1) that proprietary API-based models such as GPT4V and GemoniPro-V consistently outperform prior open-source LVLMs in various multimodal capabilities including visual perception, visual knowledge acquisition, visual reasoning, visual commonsense, and embodied intelligence, but still suffer from hallucination issues. The overall performance listed in

- Fig. 1 presents the challenging nature of our benchmark. For example, the advanced GPT4V only attains a 67.7/100 overall score on our TinyLVLM-eHub, indicating plenty of room for improvement. Besides the above comprehensive quantitative evaluation of LVLMs, we also present a productivity test for top-performing LVLMs with various demos. By evaluating Bard, Claude3-Opus, GeminiPro-V, and GPT4V as education assistants, document analysts, GUI navigators, and code generators, we find that while LVLMs show promise in understanding elements in simpler scenarios, they encounter difficulties in tasks requiring expert domain knowledge, intricate image-text alignment, precise localization, and complex layout generation.

The contributions of TinyLVLM-eHub are summarized as follows. (1) We propose a lightweight benchmark called TinyLVLM-eHub which can thoroughly evaluate various multimodal capabilities of LVLMs with only 2.1K image-text pairs. (2) We propose ChatGPT-based Ensemble Evaluation (CEE) as a judge to assess the model’s prediction and reference answer in an open-set scenario, which aligns with Human evaluation well. (3) Our comprehensive evaluation reveals that proprietary API-based models such as GPT4V and GemoniProV consistently outperform prior open-source LVLMs in various multimodal capabilities. However, they still suffer from object hallucination and encounter difficulties in tasks demanding intricate multimodal capability. We hope that our work can serve as a baseline assessment for LVLMs, and encourage further investigation on foundation multimodal models.

II. RELATED WORK A. Large Vision-Language Models

Large vision-language models (LVLMs) have achieved remarkable progress in various multimodal tasks. Owing to the development of open-source Large Language Models (LLM) such as LLaMA [11] and OPT [22], LVLMs can utilize the knowledge from LLMs and align visual features to the text space. For example, Flamingo [23] pioneers to insert cross-attention layers into LLMs to import visual features. To

TABLE I: The comparison between TinyLVLM-eHub and prior evaluation benchmarks. ‘Cap’, ‘MC’, and ‘Open’ indicate tested multimodal capability, multi-choice and free-form answers, respectively. We can see that TinyLVLM-eHub assess various multimodal capabilities absorbed from massive tasks with only 2.1K test samples through GPT-based evaluation techniques. ‘WM’, ‘Prefix’, and ‘GPT’ denote word-matching, prefix-based and ChatGPT-based evaluation methods, respectively.

Benchmark # Sample # Cap # Task Answer Eval POPE [29] 6K 1 3 MC WM

ImageNetVC [30] 4K 1 5 MC Prefix OCRBench [21] 1K 1 29 Open WM

GVT [31] 505K 4 4 MC/Open WM MMBench [19] 3K 2 20 MC WM/GPT

LVLM-eHub [16] 393K 5 42 MC/Open WM TinyLVLM-eHub 2.1K 6 42 Open GPT

further extract effective visual prompts from images, BLIP2 [8] incorporates a pre-trained visual encoder with frozen LLM by a Q-Former. Motivated by the great success of the instruction-tuning pipeline in enhancing LLMs, recent work fine-tunes LVLMs with amounts of instruction-following data. For instance, LLaVA [3] constructs 158K multimodal language-image instruction-following data to train adaption parameters and LLM. Due to the great success, LLaVA-158K instruction following data are utilized in various LVLMs such as mPLUG-owl [24], LLaMA-Adapter V2 [4], Otter [2] and Otter-I [17]. Moreover, MiniGPT-4 [1] develops a high-quality and well-aligned instruction dataset to train one projection layer, exhibiting many multimodal capabilities. Built upon MiniGPT4, VPGTrans [25] employs a technique to transfer the text encoder of a BLIP2 model to Vicuna, which reduces training costs remarkably. OF-V2 builds upon advanced LLMs and exhibits good performance on many VQA tasks. Furthermore, PandaGPT [26] can take multimodal inputs such as image, sound, and video simultaneously and compose their semantics naturally to perform complex tasks such as detailed image description generation. In addition, the closed-source APIbased models such as Bard [27], GeminiPro-V [14], GPT4V [13], and Claude3-Opus [28] distinguish themselves with their exceptional multimodal capabilities. In this work, we develop an evaluation suit to assess how well these LVLMs perform in various multimodal tasks.

B. Evaluation of Large Vision-Language Models

Lots of research activities focus on evaluating LVLMs’ capabilities, which helps understand their strengths and weaknesses and guides the further development of LVLMs. For example, Li et al. [29] present a systematic investigation of object hallucination of LVLMs by proposing a polling-based object probing evaluation method. Moreover, ImageNetVC [30] studies how well current LVLMs can master visual commonsense knowledge. Liu et al. [21] comprehensively evaluate the performance of LVLMs in visual recognition with text recognition such as Optical Character Recognition (OCR). GVT [31] evaluates LVLM’s visual semantic understanding and fine-grained perception capabilities. However, these studies only evaluate specific tasks with a portion of LVLMs, lacking

[Figure 2]

- Fig. 2: Visualization of TinyLVLM-eHub. (a) shows that TinyLVLM-eHub consists of 16 representative models including proprietary API-based LVLMs such as GPT4V. (b) presents six categories of capabilities tested in our TinyLVLM-eHub.

an overall understanding of LVLMs’ capabilities. Concurrent with our work, recent benchmarks [16], [18], [19], [32] assess LVLMs’ multimodal capability by experimenting with amounts of vision-language samples. The difference is that our benchmark aims to measure LVLM’s performance on various multimodal tasks with high efficiency. As shown in Table I, our TinyLVLM-eHub assess various multimodal capabilities absorbed from massive tasks with only 2.1K test samples. Other than quantitative evaluation, TinyLVLM-eHub also provide various demos to compare several advanced LVLMs including Bard, GPT4V, GeminiPro-V-V, and Claude3-Opus.

III. TINY LVLM EVALUATION HUB

In this section, we introduce our TinyLVLM-eHub, including an LVLM hub, the multimodal capability of interest, and the evaluation method. Compared with LVLM-eHub [16], the tiny version in this work contains more LVLM models, a more lightweight sample suite, and a more accurate evaluation technique. The overall paradigm of TinyLVLM-eHub is illustrated in Fig. 2.

A. Model Hub

We construct an LVLM model hub by collecting 16 representative LVLM models. As shown in Fig. 2, our LVLM model hub consists of BLIP2 [8], InstructBLIP [9], LLaVa [33], LLaMA-Adapter V2 [4], MiniGPT-4 [1], mPLUG-Owl [24], OF-V2 [34], Otter [2], Otter-I [17], PandaGPT [26], VPGTrans [25]. Moreover, we also include prior multimodal model OFA+ [35] and API-based LVLMs including Bard [27], GeminiPro-V [14], GPT4V [13], and Claude3-Opus [28]. The descriptions of model details have been presented in Section II. For more information, readers are suggested to refer to their original papers. It is important to note that our access to APIbased models is limited to online functionality, and we do not

possess information regarding the specific configurations of these models. As observed in LLM [11], [22], the performance of an LVLM is heavily influenced by its parameter size. For comparison purposes, all the above LVLMs have parameter sizes less than 10B except for API-based models.

B. Multimodal Capability

Capability Dimension. Following LVLM-eHub [16], we evaluate LVLMs’ capability from six aspects, including visual perception, visual knowledge acquisition, visual reasoning, visual commonsense, object hallucination, and embodied intelligence. Visual perception and visual knowledge acquisition are used to detect vision ability, where visual perception like image classification is the ability to recognize the scene or objects in images while visual knowledge acquisition such as OCR needs to understand images beyond perception for knowledge acquisition. Vision Reasoning is used to assess multimodal ability, which requires a common understanding of the relationship between text as well as pictures. Moreover, the visual commonsense aims to measure the model’s comprehension of commonly shared human knowledge about generic visual concepts. Object hallucination, which is a common issue in large foundation models, measures whether the LVLM can determine the existence of objects for given images. Lastly, embodied intelligence tests the effectiveness of guiding the agent to complete a series of tasks.

Capability Decomposition. Fig. 2 provides an overview of the evaluation process for each multimodal capability, as demonstrated through the collection of tasks and benchmarks. This involves leveraging tasks such as Image Classification (ImgCls), Object Counting (OC), and Multi-Class Identification (MCI) to evaluate the ability of visual perception. Similarly, tasks such as Optical Character Recognition (OCR) and Key Information Extraction (KIE) are utilized to evaluate

[Figure 3]

- Fig. 3: Two evaluation cases where word matching fails but CEE succeeds. In the upper one, the model’s output is selfcontradictory and includes all possible options where the word matching evaluation can cheat. For the bottom sample, the model generates a different expression (i.e., paraphrasing in general) from the ground truth. While they essentially have the same meaning, word matching definitely fails.

the ability of visual knowledge acquisition, while tasks of visual question answering and Knowledge-Grounded Image Description (KGID) are employed to evaluate the ability of visual reasoning. Furthermore, the dataset of ImageNetVC is used to evaluate the ability of commonsense, while the POPE [29] pipeline is utilized to evaluate the degree of object hallucination. Finally, the benchmark of Virtual Home is utilized to evaluate the ability of embodied intelligence. The evaluation details are presented in Section IV.

Data Collection. We investigate the aforementioned multimodal capabilities by collecting 42 standard text-related visual benchmarks. To create a lightweight evaluation suite, we have restricted each dataset to a total of 50 records except that Virtual Home [36] in embodied intelligence has six pieces of data for efficient human annotation. Therefore, it is convenient to test various LVLMs under our TinyLVLM-eHub. As a precautionary measure, we have filtered out images that contain human portraits, vulgar content, and medical organs. This ensures that API-based LVLMs can produce results without encountering any warning messages.

C. Evaluation Method

We use zero-shot evaluation to assess LVLMs. To this end, LVLM-eHub [16] assesses the predicted answer and reference answer by word matching (i.e., the prediction is correct as long as it exists in the reference answer). However, simple word matching is not effective in comparing answer pairs as illustrated in Fig. 3. Although recent works [18], [19] assess the predicted answer and the reference answer by constraining

the model’s output to be fixed forms (e.g. Yes/No or A/B/C/D), they fail to evaluate LVLMs’ capability in the open-set setting.

To tackle the above drawbacks, we introduce a new evaluation metric called ChatGPT Ensemble Evaluation (CEE) which consists of a diverse prompt generation and an ensemble voting, as shown in Fig. 4. Specifically, CEE first customizes two rules for prompt generations. For each rule, several prompts are generated by GPT-4. Given each tuple of (prompt, question, reference answer, predicted answer), ChatGPT is employed to determine whether the predicted answer is correct, which is a promising assessment [37]. Due to a good understanding of the texts of ChatGPT, CEE allows for free-form predicted answers. Finally, CEE votes for the final answer by collecting all judgments from ChatGPT. With diverse prompts ensemble, CEE can be robust and accurate in evaluating the performance of LVLMs. An example of prompt and ChatGPT’s judgment is given in Fig. 4. We show that our CEE can align with Human’s evaluation better than the word matching by experiments in Section IV-B.

IV. EXPERIMENT AND ANALYSIS

This section focuses on evaluating the performance of LVLMs in various areas such as visual perception, visual knowledge acquisition, visual reasoning, visual commonsense understanding, visual object hallucination, and embodied intelligence, as detailed in Sections IV-A. Additionally, we provide an ablation study for the benchmark in Section IV-B. Lastly, we present multimodal applications of top-performing LVLMs by productivity test in Section IV-C. The summarized results are reported in Fig. 1.

[Figure 4]

Fig. 4: Illustration of our proposed evaluation methods.

- TABLE II: Evaluation results of visual perception capability of LVLMs on tasks of Image Classification (ImgCls), Object Counting (OC), and Multi-class Identification (MCI). The accuracy is used to measure the performance of all datasets. The best result is in bold. The average score means the accuracy over all datasets on average.

ImgCls OC [15] MCI [15]

Model

Avg. Score

INet1K [38] CIFAR10 [39] Pets37 [40] FLowers102 [41] COCO [42] VCR [43] COCO VCR

OFA+ 14 42 12 2 62 74 42 78 40.8 BLIP2 68 86 40 38 48 20 88 88 59.5

InstructBLIP 64 90 30 46 62 30 78 66 58.3 LA-V2 54 74 40 34 48 30 70 64 51.8 LLaVA 44 86 20 18 34 30 48 46 40.8

MiniGPT-4 52 64 28 36 38 18 48 46 41.3 mPLUG-Owl 42 72 56 38 32 18 56 42 44.5

OF-V2 42 82 60 42 6 12 50 46 42.5 Otter 22 52 20 14 46 30 64 44 36.5

Otter-I 50 82 8 10 56 30 48 44 41.0 PandaGPT 44 76 8 2 36 34 70 60 41.3 VPGTrans 54 92 30 28 38 18 32 52 43.0

Bard 58 46 58 40 60 22 72 80 54.5 GeminiPro-V 38 60 80 52 62 90 36 88 63.3

GPT4V 50 54 84 76 62 94 30 88 67.2 Claude3-Opus 54 26 84 80 28 82 12 78 55.5

- A. Evaluation Results on Multimodal Capability

- 1) Results on Visual Perception.: We evaluate the ability of visual perception through tasks of ImgCls, MCI, and OC. The results are reported in Table II. It can be seen that GPT4V outperforms other LVLMs by a large margin on average. OFA+ achieves the worst performance in visual perception. It indicates that the fashion of the clip encoder and large language model used in current LVLMs is superior to the previous model architecture in OFA+. In addition, API-based LVLMs exhibit evident advantages over other LVLMs on the task of object counting and multi-class identification. However, the performance of these models on CIFAR10 lags behind open-source LVLMs. It may be caused by the low resolution of images in CIFAR10.
- 2) Results on Visual Knowledge Acquisition: We evaluate the ability in visual knowledge acquisition through tasks of OCR and KIE. The results are shown in Table III. We can see that API-based models still outperform other LVLMs by a large margin. Note that these models achieve remarkable success on the KIE task compared with other LVLMs, implying that API-based models can recognize characters and aggregate useful information in the image. From Table III, Claude3-Opus consistently achieves strong performance on various OCR and KIE tasks, indicating that Claude3-Opus is good at acquiring detailed information in the image.

- 3) Results on Visual Reasoning.: We evaluate the ability in visual reasoning on tasks of VQA and KGID. The results are shown in Table IV. We draw several conclusions. First, GPT4V achieves the best performance in visual reasoning compared with other LVLMs. It shows that GPT4V has a comprehensive understanding of images and related texts. Second, API=based models obtain less competitive performance than BLIP2 on WHOOPS whose VQA samples are created by breaking commonsense, implying that the commonsense understanding can be further improved. Third, API-based models also have a good understanding of science knowledge because they achieve good performance in ScienceQA which denotes the questionanswering pairs with visual inputs in ScienceQA [66].
- 4) Results on Visual Commonsense.: We perform the visual commonsense evaluation on the ImageNetVC dataset which asks the common concepts of visual input, such as color, shape, material, component, and others. Table V presents the results. We see that GPT4v and Claude3-Opus present good performance on this task. In particular, Claude3-Opus exhibits a good understanding of all aspects while GPTV lags in object materials on ImageNetVC. Bard has a good understanding of commonsense but leaves room for improvement. Specifically, we can see from Table V that Bard does not well comprehend the commonsense related to color, shape, and material compared with InstructBLIP [9].

- TABLE III: Comparison of Zero-shot Performance for Large-scale Vision and Language Models (LVLMs) on OCR and KIE Tasks.

Model

OCR KIE

Avg. Score

IIIT5K [44] IC13 [45] IC15 [46] Total-Text [47] CUTE80 [48] SVT [49] SVTP [50] COCO-Text [51] WordArt [52] CTW [53] HOST [54] WOST [54] SROIE [55] FUNSD [56] OFA+ 0 2 0 4 0 2 0 2 0 0 0 0 2 0 0.9 BLIP2 68 86 52 58 72 76 72 48 40 54 50 60 0.0 4.0 52.9

InstructBLIP 82 86 66 62 78 76 74 56 48 56 54 56 0.0 4.0 57.0 LA-V2 52 20 28 28 28 20 26 20 36 20 16 6.0 4.0 56 25.7 LLaVA 14 8.0 14 12 24 2.0 10 14 26 12 14 8.0 2.0 42 14.4

MiniGPT-4 20 18 8.0 6.0 16 10 6.0 12 20 6.0 6.0 4.0 2.0 20 11.0 mPLUG-Owl 22 4.0 20 14 24 4.0 8.0 8.0 24 8.0 6.0 2.0 0.0 16 11.4

OF-V2 28 18 24 20 22 10 22 18 28 16 14 6.0 0.0 28 18.1 Otter 4.0 6.0 8.0 10 6.0 6.0 4.0 6.0 8.0 4.0 2.0 4.0 0.0 26 6.7 Otter-I 18 6.0 18 24 16 12 12 16 26 8.0 14 2.0 2.0 38 15.1 PandaGPT 2.0 0.0 0.0 4.0 4.0 0.0 0.0 4.0 0.0 2.0 0.0 0.0 0.0 22 2.7 VPGTrans 50 80 36 56 42 64 62 32 42 44 46 42 0.0 22 44.1

Bard 78 84 60 60 76 74 66 42 54 64 46 52 42 50 60.6 GeminiPro-V 86 90 52 60 72 82 60 32 68 48 46 40 60 46 60.1

GPT4V 78 70 42 54 72 44 32 38 74 58 28 38 78 86 56.6 Claude3-Opus 86 86 70 60 84 80 74 32 66 60 48 52 74 58 66.4

- TABLE IV: Comparison of Zero-shot Performance for LVLM Models on VQA and KGID Tasks. In these experiments, top-1 accuracy is employed for evaluation.

Model

VQA KGID

Avg. Score

DocVQA [57] TextVQA [58] STVQA [59] OCR-VQA [60] OKVQA [61] GQA [62] IconQA [63] VSR [64] WHOOPS [65] ScienceQA [66] VizWiz [67]

OFA+ 2 0 16 20 22 50 16 44 8 4 28 19.1 BLIP2 6.0 36 40 52 52 36 46 66 56 66 66 47.5

InstructBLIP 10 40 52 76 66 58 42 54 42 48 78 51.5 LA-V2 20 54 58 50 58 44 44 52 40 56 54 48.2 LLaVA 8.0 34 42 34 34 44 40 52 30 54 64 39.6

MiniGPT-4 12 34 30 34 36 20 32 48 22 6.0 38 28.4 mPLUG-Owl 2.0 28 26 18 16 20 22 46 12 10 26 20.5

OF-V2 8.0 34 52 44 34 40 48 58 32 48 58 41.5 Otter 10 24 30 28 54 20 34 24 12 34 46 28.7

Otter-I 14 40 46 34 50 44 36 56 20 48 54 40.2 PandaGPT 10 16 24 30 48 38 34 60 14 50 42 33.3 VPGTrans 22 38 42 32 36 34 32 40 36 12 48 33.8

Bard 48 60 72 80 68 40 62 82 42 68 62 62.2 GeminiPro-V 56 66 78 62 36 38 58 62 40 74 50 56.4

GPT4V 76 84 66 72 58 36 76 72 30 74 66 64.5 Claude3-Opus 86 80 64 64 50 24 52 52 42 82 68 60.4

- TABLE V: Comparisons of Zero-shot visual commonsense Performance for LVLM Models on ImageNetVC datasets. Top1 accuracy is employed for the evaluation.

hallucination is different from the type of other LVLMs which tends to answer ‘yes’ even when the object does not exist in the image. To our surprise, OFA+ achieves much better performance on object hallucination than many LVLMs. This indicates that the hallucination problem might come from the LLM decoder in LVLMs. Our experiment reveals that the object hallucination issue of LVLMs still remains a challenging problem.

ImageNetVC [30]

Model

Avg. Score

Color Shape Material Component Others

OFA+ 54 28 38 48 57 45.2 BLIP2 32 16 36 76 66 45.2

InstructBLIP 52 58 64 76 76 65.2 LA-V2 42 38 62 76 72 58.0 LLaVA 42 38 50 50 54 46.8

MiniGPT-4 30 28 36 50 32 35.2 mPLUG-Owl 14 16 34 26 28 23.6

6) Results on Embodied Intelligence.: We present the evaluation results on embodied intelligence. Similar to LVLMeHub [16], we conducted a user study involving 10 participants to assess the effectiveness of planning outputs. The study comprises 6 household scenarios from VirtualHome [36]. The results are reported in Table VII. Given that Bard is specifically designed for images devoid of human presence, we present evaluation results for Bard on test splits both with and without human subjects. Bard (w/o human) garners the highest average score across five key dimensions and exhibits unmatched performance in terms of reasonability and executability. However, its proficiency in Object Recognition fails to rank among the top three, highlighting limitations in its fine-grained visual detection capability within the embodied domain. Moreover, acknowledging the frequent occurrence of human presence, and even direct human interaction in daily embodied tasks, it’s evident that Bard has considerable room for improvement. In addition, we see that GminiPro-V achieves the second-best performance but attains a low score in conciseness. Striking a balance between maintaining human preference and ensuring task efficiency presents a substantial development frontier for API-based models in the embodied domain tasks.

OF-V2 44 32 48 56 48 45.6 Otter 36 30 44 52 64 45.2

Otter-I 46 40 54 60 64 52.8 PandaGPT 48 34 48 64 58 50.4 VPGTrans 36 48 46 70 48 49.6

Bard 40 44 52 82 72 58.0 GeminiPro-V 56 62 50 82 60 62.0

GPT4V 54 66 58 90 82 70.0 Claude3-Opus 52 68 70 86 80 71.2

5) Results on Object Hallucination.: We test the degree of object hallucination of Bard on MSCOCO under the POPE framework [29] which asks YES/NO questions about the existence of objects given an image. We report results in terms of accuracy, precision, recall, and Yes (the ratio of answering Yes). The results are shown in Table VI. We can see that API-based models achieve less satisfactory performance than the other 2 LVLM models, including InstructBLIP, and BLIP, showing that they still suffer from object hallucination. By comparing the results of precision, recall, and yes, we find that Bard tends to stick in the mud and often answers ‘no’ even when the object indeed exists in the image. Such object

- TABLE VI: Detailed evaluation results of the zero-shot performance of LVLMs on MSCOCO using POPE evaluation pipeline [29], where Acc represents the accuracy of prediction; Prec represents how many of the predicted positive samples are true positive samples; Recall represents how many of all true positive samples are correctly identified; and Yes represents the probability that the model outputs a yes answer. The average score is calculated based on the metric of accuracy.

Model

MSCOCO-Random [29] MSCOCO-Popular [29] MSCOCO-Adversarial [29]

Avg. Score

Acc Prec Recall Yes Acc Prec Recall Yes Acc Prec Recall Yes

OFA+ 80 100 75 44 80 100 84 42 76 100 84 42 78.7 BLIP2 72 100 52 30 86 87 86 48 90 95 84 44 82.7

InstructBLIP 82 100 81 39 92 92 92 49 92 92 92 49 88.7 LA-V2 64 59 76 74 46 43 72 84 46 45 80 88 52.0 LLaVA 54 54 93 100 46 46 92 100 40 40 80 100 46.7

MiniGPT-4 65 71 71 60 63 71 74 56 46 44 53 61 58.0 mPLUG-Owl 68 63 100 86 66 61 95 88 59 60 86 81 64.3

OF-V2 54 55 93 98 48 49 96 98 50 50 96 96 50.7 Otter 47 44 71 92 42 42 84 100 44 44 84 96 44.3

Otter-I 76 88 76 50 68 69 88 64 66 66 92 70 70.0 PandaGPT 58 56 93 96 50 48 92 96 48 46 88 96 52.0 VPGTrans 67 92 46 28 80 94 65 35 79 88 65 36 75.3

Bard 63 100 36 18 70 100 40 18 69 100 43 19 67.3 GeminiPro-V 70 94 52 32 86 87 84 48 84 90 76 42 80.0

GPT4V 70 89 55 36 86 84 88 52 74 70 84 60 76.7 Claude3-Opus 68 84 55 38 80 82 76 46 70 69 72 52 72.7

- TABLE VII: Generated planning quality evaluation on embodied tasks. 10 participants are involved in the user study for evaluation. The evaluation comprises five dimensions with scores ranging from 1 to 5, including object recognition (OR), spatial relationship (SR), level of conciseness (Con.), reasonability (Rea.), and executability (Exe.) of the planning. The final score for each dimension is averaged over all participants and normalized by (·)/5 × 100%. Bard∗ means that only samples without describing humans are included. We see that Bard exhibits good planning ability for embodied application.

telligence. We verify the representativeness of TinyLVLM-eHub by calculating Pearson’s correlation between score sequences of LVLMs existing in both two benchmarks (BLIP2, LA-V2, LLaVA, MiniGPT-4, mPLUG-Owl, and Otter). We collect Pearson’s correlations on overall score and five multimodal capabilities including visual perception, reasoning, knowledge acquisition, commonsense, and hallucination. As shown in Table VIII, the overall score of LVLMs in TinyLVLM-eHub is highly correlated (0.98) with that in LVLM-eHub. The correlation score on each capability is more than 0.8. These results validate the efficacy of the TinyLVLM-eHub as a proxy for the comprehensive LVLM-eHub, thus ensuring that the lightweight benchmark still provides a robust indication of model performance. The lightweight nature of TinyLVLMeHub will be accessible to a wider range of participants.

Virtual Home [36]

Model

Avg. Score

OR SR Con. Rea. Exe.

OFA+ 22.5 21.7 49.2 20.8 20.0 26.8 BLIP2 40.6 33.6 65.0 55.6 57.6 50.4

InstructBLIP 61.6 55.6 49.6 64.0 62.0 58.6 LA-V2 76.2 74.2 59.2 80.8 81.6 74.4 LLaVA 77.6 72.2 37.2 74.0 76.4 67.4

TABLE VIII: The Pearson’s correlations between the score sequences of LVLMs existing both in TinyLVLM-eHub and LVLM-eHub are calculated. The results are obtained on the overall score and scores on five multimodal capabilities.

MiniGPT-4 74.0 69.4 32.4 70.8 62.2 61.8 mPLUG-Owl 68.4 64.4 29.6 68.8 70.8 60.4

OF-V2 23.2 24.2 77.2 37.0 35.8 39.4

Otter 67.6 62.0 37.2 61.4 62.4 58.2 Otter-I 81.0 85.0 57.8 76.4 80.0 76.0

Overall Score Perception Knowledge Reasoning Commonsense Hallucination Pearson’s r 0.98 0.86 0.99 0.98 0.87 0.80

PandarGPT 74.8 74.6 65.8 89.4 89.4 78.8 VPGTrans 68.6 64.4 35.2 67.0 67.0 60.4

2) Compared with word-matching evaluation method: We ablate evaluation methods of word matching and CEE in terms of agreement with human evaluation in Table IX. Among all datasets studied in this work, we manually select 5 representative and diverse ones, namely IC15, ImageNetVC shape, MSCOCO POPE adversarial, VCR MCI, and VSR, to conduct the human evaluation. As illustrated in Table IX, 11 out of 12 models show noticeably better agreement (i.e., accuracy averaged over all samples of 5 datasets) of CEE than word matching with human annotation, while both methods enjoy a high agreement of greater than 80%.

Bard∗ 73.0 79.8 79.0 94.2 91.8 83.6 GeminiPro-V 88.3 86.7 31.7 87.5 89.2 76.7

GPT4V 52 68 70 86 80 71.2 Claude3-Opus 54 66 58 90 82 70.0

- B. Ablation Study

1) The representativeness of TinyLVLM-eHub: Note that each dataset of TinyLVLM-eHub consists of 50 samples from the original LVLM-eHub [16] except for tasks in embodied in-

TABLE IX: The comparison of the alignment with human evaluation between the word matching approach [16] and our ChatGPT Ensemble Evaluation (CEE). Higher alignment indicates more consistent evaluation with human annotation. We see that CEE achieves higher alignment on all LVLMs except for Bard and aligns generally better with human evaluation when more prompts are employed for ensemble evaluation.

Evaluation BLIP2 InstructBLIP LA-V2 LLaVA MiniGPT-4 mPLUG-Owl OF-V2 Otter Otter-I PandaGPT VPGTrans Bard Avg Word Matching [16] 85.0 86.0 90.0 85.2 85.6 87.6 83.2 80.8 92.0 82.4 85.6 92.0 86.3

- CEE (3 prompts) 87.1 89.6 88.6 89.6 86.4 88 88.2 83.2 92.8 88.2 88.6 88.4 88.2

- CEE (4 prompts) 88.0 90.4 88.8 88.8 86.8 88.4 89.6 83.6 92.8 88.0 88.0 85.8 88.3

- CEE (5 prompts ours) 89.2 90.0 90.8 89.6 87.6 90.0 90.8 82.4 92.8 88.0 87.6 86.4 88.9

[Figure 5]

Fig. 5: The sensitivity of ChatGPT Ensemble Evaluation (CEE) to random noise by visualizing the heatmap of score variances across 16 LVLMs. These results are based on evaluations conducted on 5 datasets with three independent evaluations each. Our findings indicate that CEE evaluation results demonstrate robustness to random noise, as evidenced by the small variance observed across multiple evaluation processes.

Bard is the only exception where CEE performs worse than word matching. LLMs are well known for being capable of generating long, complex, and coherent text. Bard is much more talkative than others and hence more likely inclined to output verbose responses. Therefore, from another perspective, Bard is also more competent in fabricating unfaithful and/or nonfactual statements or reasoning. Besides, due to its close source, we have no explicit or guaranteed way to control its generation length as we can do to open source LVLMs (i.e., max_new_tokens). Empirically, while Bard indeed can follow the specified answer format in the prompt, it continues generation after formatted answers and usually ends with irrelevant messages. Based on the observations above, we hypothesize that Bard’s coherent but talkative responses with possible hallucinations could hurt CEE, especially when ground-truth answers of those 5 chosen datasets are all succinct, close-set, and definite.

- 3) The effect of the number of prompts used in CEE: We

investigate the effect of the number of prompts used in the proposed CEE. Specifically, we use CEE with 3/4/5 prompts ensemble to evaluate LVLMs and report the alignment with human evaluation. As shown in Table IX, CEE generally aligns better with human evaluation when more prompts are employed for ensemble evaluation.

- 4) The sensitivity of CEE to randomness: We assess the

sensitivity of ChatGPT Ensemble Evaluation (CEE) to random noise by visualizing the heatmap of score variances across 16 LVLMs. These results are based on evaluations conducted on 5 datasets (IC15, ImageNetVC shape, MSCOCO POPE adversarial, VCR MCI, and VSR) with three independent evaluations each. The results are reported in Fig. 6 where we can

[Figure 6]

Fig. 6: The sensitivity of CEE to different LLMs. We report the correlation between overall scores of 16 LVLMs using the proposed CEE evaluation with different LLMs as judges. The LLM judges comprise ChatGPT, GPT4, and language-only GeminiPro. The overall score sequences obtained by different LLMs are highly correlated with each other.

observe the small variance across multiple evaluation processes. Specifically, the average variance across all 16 LVLMs of IC15, ImageNetVC shape, MSCOCO POPE adversarial, VCR MCI, and VSR are 1.39, 1.20, 1.29, 1.04, 1.15, respectively. Our experiments indicate that CEE evaluation results are robust to random noise.

5) The effect of different LLMs on CEE: We investigate the sensitivity of CEE to different LLM judges including ChatGPT (used by our CEE), GPT4, and language-only GeminiPro. To explore the influence of CEE with different LLM judges, we calculate the correlation between overall scores of 16 LVLMs using the proposed CEE evaluation. The results are reported in Fig. 5 where we see that the overall score sequences obtained by different LLMs are highly correlated with each other. It implies that the overall scores on TinyLVLM-eHub assessed by different LLMs are consistent with each other.

C. Productivity Test of Bard, Claude3-Opus, GeminiPro-V, and GPT4V

In this section, we further assess the productivity of topperforming LVLMs, i.e. Bard, Claude3-Opus, GeminiPro-V, and GPT4V). To this end, four types of multimodal applications are presented in Section IV-C1 to Section IV-C4 including education assistant, document analyst, GUI navigator, and code generator.

1) Education Assistant: We focus on LVLMs’ ability as an education assistant to solve discipline VQAs. Firstly, we feed strong LVLMs with an image of the food chain and ask them to identify which animals are on the top of the food chain. As shown in Fig. 7, Bard, Claude3-Opus, and GPT4V identify the hawk and mountain lion as the apex predators at the top of the food chain. They also provide correct reasoning steps that these animals are depicted without any natural predators in the image, and point out the arrows indicating their position at

[Figure 7]

- Fig. 7: Education assistant case study I. Strong LVLMs understand the complex food chain well. Bard, Claude3-Opus, and GPT4V can answer the question correctly while GeminiPro-V only provides one of the answers.

[Figure 8]

- Fig. 8: Education assistant case study II. Strong LVLMs struggle to effectively grasp and apply specialized physics knowledge.

the top of the food web. While all three models agree on the answer, GeminiPro-V focuses solely on the mountain lion. It reasons that since the mountain lion is a predator and has no predators listed in the food chain, it must be at the top. This explanation considers only the mountain lion, not the hawk. Overall, strong LVLMs can understand the complex food chain well with minor flaws.

Secondly, LVLMs were evaluated using a physics problem concerning light propagation in a middle school context. As shown in Fig. 8, five diagrams were presented, and LVLMs were tasked with identifying the one that does not

depict a plausible path for a light ray transitioning from glass to air. Only GeminiPro-V provided the correct answer. However, it erroneously stated that in diagram D, the light ray bends away from the normal upon exiting the glass block, indicating flawed reasoning. In reality, when light transitions from a denser medium (glass) to a less dense one (air), it bends away from the normal. The other models incorrectly identified answer E. Bard demonstrated a misunderstanding of light propagation laws, while Claude3-Opus presented an inaccurate law stating that light bends towards the normal when transitioning from a higher to a lower refractive index

[Figure 9]

- Fig. 9: Document Analyst case study I. Strong LVLMs can understand elements and their relationships in the chart but struggle to extract numbers accurately.

[Figure 10]

Fig. 10: Document Analyst case study II. Strong LVLMs can understand and extract elements in the table image.

medium. Only GPT4V correctly comprehended the principle of refraction. Consequently, even strong LVLMs struggle to effectively grasp and apply specialized physics knowledge.

2) Document Analyst: Document understanding enhances the LVLM’s ability to assist users, perform complex tasks, and contribute effectively in various domains such as education, research, customer service, and more.

We proceed with a visual question on the chart image. As shown in Fig. 9, the image depicts the accuracy of three models in five datasets. LVLMs are asked to find which model achieves the highest accuracy in Dataset 3. From responses, we see that Bard, Claude3-Opus and GeminiPro-V can identify that model 1 has the highest accuracy in Dataset 3. However, Claude3Opus does not provide reasoning details about the result and Bard and GeminiPro-V cannot extract the point in the chart accurately. In addition, GPT4V even gives the wrong output. These responses show that Strong LVLMs can understand elements and their relationships in the chart but struggle to

extract numbers accurately.

We then provide these LVLMs with an easier case in document understanding. As shown in Fig. 10, LVLMs are required to find the number of medals won by Robert in total with a table of medals won by different players provided. The table QA is easier than chart QA in Fig. 9 because the correspondence between elements is shown in the structured table. As we see, Bard, GenminiPro-V, and GPT4V can answer the question correctly with the number of gold, silver, and bronze medals presented. Only Claude3-Opus recognizes the number of silver medals won by Robert inaccurately.

In conclusion, the evaluation highlights that while Strong LVLMs exhibit varying degrees of proficiency in document understanding tasks, challenges persist in accurately extracting and interpreting information, particularly from visual elements such as charts. While Bard, Claude3-Opus, and GeminiProV demonstrate the ability to identify relationships within the chart data, shortcomings in precise numerical extraction are

[Figure 11]

- Fig. 11: GUI Navigator case study I. Strong LVLMs can understand users’ requests and the user interface of the phone.

[Figure 12]

- Fig. 12: GUI Navigator case study II. Strong LVLMs can find the app icon for the user’s request but cannot localize the app accurately.

evident across all models. Conversely, structured data in table format, as seen in the easier case, enables more accurate responses from Bard, GeminiPro-V, and GPT4V, indicating that structured presentation facilitates comprehension and extraction of relevant details. These findings underscore the importance of ongoing research and development efforts to enhance LVLMs’ proficiency in document understanding, particularly in handling diverse data formats and extracting precise information for improved performance across various tasks and domains.

3) GUI Navigator: GUI navigation enables users to interact with LVLMs in an intuitive and user-friendly manner. Users can

input queries, navigate through options, and visualize results more effectively through an LVLM agent. We assess whether strong LVLMs can be used as a good GUI navigator. Note that Bard is not included in the test because Bard is not accessible now.

Firstly, we feed LVLMs with a phone screenshot, and a search bar open on what appears to be the YouTube mobile app. The question is ”Given the image, what should I do next to learn how to bake a chocolate cake?” As shown in Fig. 11, Bard, Claude2-Opus, and GPT4V provide correct guidance. They accurately identify the search bar in the image and suggest

[Figure 13]

- Fig. 13: Code generator case study I. Strong LVLMs can generate code for user’s request. GPT4V performs best because the generation of HTML code almost matches the layout of the user’s draft.

using it to search for a chocolate cake recipe. While the level of detail provided differs. Claude3-Opus mentions following video tutorials, while GeminiPro-V emphasizes reading the recipe instructions. All the responses effectively address the user’s query.

We proceed with a more difficult navigation task requiring fine-grained localization capability. As shown in Fig. 12, Claude3-Opus and GPT4V analyze the screenshot correctly to answer the question about which app should be launched to send the email. But they give the wrong answer about where to click to launch the email app. GeminiPro-V even does not recognize the Gmail App in the image. Specifically, while Claude3-Opus identified the Gmail icon and its location in the bottom dock, it incorrectly provided coordinates to click on. The provided coordinates ([135, 2310]) do not point to the email app. GPT4V also faces the same issue.

In essence, while LVLMs show promise in understanding and responding to GUI elements in simpler scenarios, they encounter difficulties in tasks demanding precise localization and action identification within graphical interfaces. Continued research and refinement are necessary to enhance LVLMs’ capabilities for seamless and accurate GUI navigation across diverse contexts and tasks.

4) Code Generator: Code generation enables translating high-level instructions or drafts into executable code, making them applicable across a wide range of domains such as software development, robotics, and automation. We evaluate the capability of code generation of LVLMs with a draft of the webpage.

As demonstrated in Fig. 13, all models can generate HTML code as required. The corresponding webpages are shown at the bottom of Fig. 13. We see that GPT4V performs best because

the generation HTML code almost matches the layout of the user’s draft. While the other three models can only generate simple HTML syntax and cannot achieve the webpage layout depicted in the sketch.

V. CONCLUSION

In this work, we propose a lightweight evaluation suite called TinyLVLM-eHub for Large Vision-Language Models (LVLMs). TinyLVLM-eHub comprehensively assesses various multimodal capabilities such as visual perception and embodied intelligence with quantitative evaluation. For a robust and accurate evaluation, we developed a new evaluation metric called CEE to assess the relationship between the reference answer and the answer predicted by LVLM. Through experiments, we demonstrate that CEE aligns better with human evaluation than the naive word match approach. By TinyLVLMeHub, we reveal that closed-source API-Based LVLMs such as GPT4V and GeminiPro-V consistently outperform open-source LVLMs in various multimodal capabilities including visual perception, visual knowledge acquisition, visual reasoning, and embodied intelligence. Through various productivity tests, we also show that while top-performing LVLMs show promise in understanding elements in simpler scenarios, they encounter difficulties in tasks requiring expert domain knowledge, intricate image-text alignment, precise localization, and complex layout generation.

Although the evaluation in TinyLVLM-eHub is comprehensive, we only assess the boundary of multimodal capability for various LVLMs. Indeed, the evaluation of LVLM must also take into account other critical factors, such as content safety, political bias, and racial discrimination. These issues have become increasingly relevant due to the potential harm

caused by biased or harmful content generated by these models. Therefore, it is crucial to thoroughly assess the ability of LVLM to produce safe and unbiased content that does not perpetuate harmful stereotypes or discriminatory attitudes. Furthermore, top-performing LVLMs have demonstrated remarkable proficiency in various multimodal capabilities, warranting a comprehensive investigation into specific aspects of their performance. Finally, TinyLVLM-eHub reveals the strengths and weaknesses of various LVLMs. Further exploration on developing LVLM should consider how to enhance the deep understanding of LVLMs in product-level scenarios.

ACKNOWLEDGMENTS

This paper is partially supported by the National Key R & D Program of China No.2022ZD0160101 & No.2022ZD0161000.

VI. REFERENCES SECTION REFERENCES

- [1] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [2] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726, 2023.
- [3] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning, 2023.
- [4] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llamaadapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010, 2023.
- [5] Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. Llavamed: Training a large language-and-vision assistant for biomedicine in one day. arXiv preprint arXiv:2306.00890, 2023.
- [6] Michael Moor, Qian Huang, Shirley Wu, Michihiro Yasunaga, Cyril Zakka, Yash Dalmia, Eduardo Pontes Reis, Pranav Rajpurkar, and Jure Leskovec. Med-flamingo: A multimodal medical few-shot learner. July

2023. arXiv:2307.15189.

- [7] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601, 2023.
- [8] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.
- [9] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023.
- [10] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023.
- [11] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, MarieAnne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [12] Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19358–19369, 2023.
- [13] OpenAI. Gpt-4 technical report, 2023.
- [14] Google. Gemini. https://gemini.google.com/app, 2024.
- [15] Guangzhi Wang, Yixiao Ge, Xiaohan Ding, Mohan Kankanhalli, and Ying Shan. What makes for good visual tokenizers for large language models? arXiv preprint arXiv:2305.12223, 2023.

- [16] Peng Xu, Wenqi Shao, Kaipeng Zhang, Peng Gao, Shuo Liu, Meng Lei, Fanqing Meng, Siyuan Huang, Yu Qiao, and Ping Luo. Lvlm-ehub: A comprehensive evaluation benchmark for large vision-language models. arXiv preprint arXiv:2306.09265, 2023.
- [17] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Mimic-it: Multi-modal in-context instruction tuning. arXiv preprint arXiv:2306.05425, 2023.
- [18] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.
- [19] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023.
- [20] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.
- [21] Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of ocr in large multimodal models. arXiv preprint arXiv:2305.07895, 2023.
- [22] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.
- [23] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for fewshot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.
- [24] Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.
- [25] Ao Zhang, Hao Fei, Yuan Yao, Wei Ji, Li Li, Zhiyuan Liu, and Tat-Seng Chua. Transfer visual prompt generator across llms. arXiv preprint arXiv:2305.01278, 2023.
- [26] Yixuan Su, Tian Lan, Huayang Li, Jialu Xu, Yan Wang, and Deng Cai. Pandagpt: One model to instruction-follow them all. arXiv preprint arXiv:2305.16355, 2023.
- [27] Google. Bard. https://bard.google.com/, 2023.
- [28] Anthropic. Claude. https://www.anthropic.com/claude, 2024.
- [29] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.
- [30] Heming Xia, Qingxiu Dong, Lei Li, Jingjing Xu, Ziwei Qin, and Zhifang Sui. Imagenetvc: Zero-shot visual commonsense evaluation on 1000 imagenet categories. arXiv preprint arXiv:2305.15028, 2023.
- [31] Guangzhi Wang, Yixiao Ge, Xiaohan Ding, Mohan Kankanhalli, and Ying Shan. What makes for good visual tokenizers for large language models? arXiv preprint arXiv:2305.12223, 2023.
- [32] Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi, Dingning Liu, Mukai Li, Lu Sheng, Lei Bai, Xiaoshui Huang, Zhiyong Wang, et al. Lamm: Language-assisted multi-modal instruction-tuning dataset, framework, and benchmark. arXiv preprint arXiv:2306.06687, 2023.
- [33] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.
- [34] Awadalla Anas and Gao Irena. Openflamingo v2, 2023.
- [35] Jinze Bai, Rui Men, Hao Yang, Xuancheng Ren, Kai Dang, Yichang Zhang, Xiaohuan Zhou, Peng Wang, Sinan Tan, An Yang, Zeyu Cui, Yu Han, Shuai Bai, Wenbin Ge, Jianxin Ma, Junyang Lin, Jingren Zhou, and Chang Zhou. Ofasys: A multi-modal multi-task learning system for building generalist models. CoRR, abs/2212.04408, 2022.
- [36] Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li, Tingwu Wang, Sanja Fidler, and Antonio Torralba. Virtualhome: Simulating household activities via programs. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8494–8502, 2018.
- [37] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685, 2023.
- [38] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International journal of computer vision, 115:211–252, 2015.

- [39] Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009.
- [40] Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman, and C. V. Jawahar. Cats and dogs. In IEEE Conference on Computer Vision and Pattern Recognition, 2012.
- [41] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In Indian Conference on Computer Vision, Graphics and Image Processing, Dec 2008.
- [42] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv preprint arXiv:1504.00325, 2015.
- [43] Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi. From recognition to cognition: Visual commonsense reasoning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6720–6731, 2019.
- [44] Anand Mishra, Karteek Alahari, and C. V. Jawahar. Top-down and bottom-up cues for scene text recognition. In 2012 IEEE Conference on Computer Vision and Pattern Recognition, pages 2687–2694, 2012.
- [45] Dimosthenis Karatzas, Faisal Shafait, Seiichi Uchida, Masakazu Iwamura, Lluis Gomez i Bigorda, Sergi Robles Mestre, Joan Mas, David Fernandez Mota, Jon Almaz`an Almaz`an, and Llu´ıs Pere de las Heras. Icdar 2013 robust reading competition. In 2013 12th International Conference on Document Analysis and Recognition, pages 1484–1493, 2013.
- [46] Dimosthenis Karatzas, Lluis Gomez-Bigorda, Anguelos Nicolaou, Suman Ghosh, Andrew Bagdanov, Masakazu Iwamura, Jiri Matas, Lukas Neumann, Vijay Ramaseshan Chandrasekhar, Shijian Lu, Faisal Shafait, Seiichi Uchida, and Ernest Valveny. Icdar 2015 competition on robust reading. In 2015 13th International Conference on Document Analysis and Recognition (ICDAR), pages 1156–1160, 2015.
- [47] Chee Kheng Ch’ng and Chee Seng Chan. Total-text: A comprehensive dataset for scene text detection and recognition. In 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR), volume 01, pages 935–942, 2017.
- [48] Anhar Risnumawan, Palaiahankote Shivakumara, Chee Seng Chan, and Chew Lim Tan. A robust arbitrary text detection system for natural scene images. Expert Systems with Applications, 41(18):8027–8048, 2014.
- [49] Cunzhao Shi, Chunheng Wang, Baihua Xiao, Song Gao, and Jinlong Hu. End-to-end scene text recognition using tree-structured models. Pattern Recognition, 47(9):2853–2866, 2014.
- [50] Trung Quy Phan, Palaiahnakote Shivakumara, Shangxuan Tian, and Chew Lim Tan. Recognizing text with perspective distortion in natural scenes. In 2013 IEEE International Conference on Computer Vision, pages 569–576, 2013.
- [51] Andreas Veit, Tomas Matera, Luk´as Neumann, Jiri Matas, and Serge J. Belongie. Coco-text: Dataset and benchmark for text detection and recognition in natural images. ArXiv, abs/1601.07140, 2016.
- [52] Xudong Xie, Ling Fu, Zhifei Zhang, Zhaowen Wang, and Xiang Bai. Toward understanding wordart: Corner-guided transformer for scene text recognition. 2022.
- [53] Yuliang Liu, Lianwen Jin, Shuaitao Zhang, Canjie Luo, and Sheng Zhang. Curved scene text detection via transverse and longitudinal sequence connection. Pattern Recogn., 90(C):337–345, jun 2019.
- [54] Yuxin Wang, Hongtao Xie, Shancheng Fang, Jing Wang, Shenggao Zhu, and Yongdong Zhang. From two to one: A new scene text recognizer with visual language modeling network. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 14194–14203, 2021.
- [55] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthenis Karatzas, Shijian Lu, and CV Jawahar. Icdar2019 competition on scanned receipt ocr and information extraction. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1516–1520. IEEE, 2019.
- [56] Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. Funsd: A dataset for form understanding in noisy scanned documents. In 2019 International Conference on Document Analysis and Recognition Workshops (ICDARW), volume 2, pages 1–6. IEEE, 2019.
- [57] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.
- [58] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 8309–8318, 2019.
- [59] Ali Furkan Biten, Rub`en Tito, Andr´es Mafla, Lluis Gomez, Marc¸al Rusi˜nol, Minesh Mathew, C.V. Jawahar, Ernest Valveny, and Dimosthenis Karatzas. Icdar 2019 competition on scene text visual question answering.

- In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 1563–1570, 2019.
- [60] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In 2019 International Conference on Document Analysis and Recognition (ICDAR), pages 947–952, 2019.
- [61] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3190–3199, 2019.
- [62] Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6693–6702, 2019.
- [63] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. Iconqa: A new benchmark for abstract diagram understanding and visual language reasoning. In The 35th Conference on Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks, 2021.
- [64] Fangyu Liu, Guy Edward Toh Emerson, and Nigel Collier. Visual spatial reasoning. Transactions of the Association for Computational Linguistics, 2023.
- [65] Nitzan Bitton-Guetta, Yonatan Bitton, Jack Hessel, Ludwig Schmidt, Yuval Elovici, Gabriel Stanovsky, and Roy Schwartz. Breaking common sense: Whoops! a vision-and-language benchmark of synthetic and compositional images. arXiv preprint arXiv:2303.07274, 2023.
- [66] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, SongChun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507– 2521, 2022.
- [67] Jeffrey P Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C Miller, Robin Miller, Aubrey Tatarowicz, Brandyn White, Samual White, et al. Vizwiz: nearly real-time answers to visual questions. In Proceedings of the 23nd annual ACM symposium on User interface software and technology, pages 333–342, 2010.

VII. BIOGRAPHY SECTION

Wenqi Shao is a Young Scientist at the Shanghai Artificial Intelligence Laboratory. He completed his PhD in 2022 at the Multimedia Lab of the Chinese University of Hong Kong (CUHK), where he was supervised by Prof. Xiaogang Wang, Prof. Ping Luo, and Prof. Hongsheng Li. Prior to his doctoral studies, he obtained a bachelor’s degree from the School of Mathematics at the University of Electronic Science and Technology of China (UESTC) in 2017. His research interests revolve around multimodal foundation models, large language model compression,

[Figure 14]

efficient transfer learning, and their applications in multimedia. Wenqi Shao is actively involved in the academic community and serves as a reviewer for several prestigious conferences, including CVPR, ICCV, ICML, NeurIPS, and ICLR.

[Figure 15]

Meng Lei is a master’s student at Peking University and received a BEng degree from The South China University of Technology. His research interest lies in computer vision and multimodal foundation models.

[Figure 16]

Yutao Hu received his B.S. degree in electronics and information engineering from Beihang University, Beijing, China in 2017, and he received his Ph.D. degree in the National Key Laboratory of CNS/ATM, School of Electronics and Information Engineering, Beihang University, Beijing, China. He is now working as a post-doctoral fellow in the University of Hong Kong. His research interests include machine learning and computer vision.

[Figure 17]

Siyuan Huang is a PhD student at Shanghai Jiao Tong University. He earned his M.Sc. degree from the Karlsruhe Institute of Technology and his B.Eng. degree from the Beijing Institute of Technology. His research interests lie in multimodal foundational models and robotics.

Peng Gao is a Young Scientist at the Shanghai Artificial Intelligence Laboratory. He completed his Ph.D. in 2021 at the Multimedia Lab of the Chinese University of Hong Kong (CUHK), where he was supervised by Prof. Xiaogang Wang and Prof. Hongsheng Li. His research interests revolve around multimodal foundation models, efficient visual backbone design, self-supervised representation learning, and their applications in multimedia. Peng Gao is actively involved in the academic community and is a reviewer for several prestigious conferences, including CVPR,

[Figure 18]

ICCV, ICML, NeurIPS, and ICLR.

[Figure 19]

Peng Xu is currently a PhD student at the University of Hong Kong. Before this, he obtained his Bachelor’s degree from the Southern University of Science and Technology, China. He is currently focusing on the inference techniques applied to large language models.

[Figure 20]

Hongsheng Li received a bachelor’s degree in automation from the East China University of Science and Technology, and master’s and doctorate degrees in computer science from Lehigh University, Pennsylvania, in 2006, 2010, and 2012, respectively. He is currently an assistant professor in the Department of Electronic Engineering at The Chinese University of Hong Kong. His research interests include computer vision, medical image analysis, and machine learning.

Yu Qiao (Senior Member, IEEE) is a professor with Shanghai AI Laboratory and the Shenzhen Institute of Advanced Technology (SIAT), Chinese Academy of Sciences. He has published more than 600 articles in international journals and conferences, including T-PAMI, IJCV, T-IP, TSP, CVPR, and ICCV. His research interests include computer vision, deep learning, and bioinformation. He received the First Prize of the Guangdong Technological Invention Award, and the Jiaxi Lv Young Researcher Award from the Chinese Academy of Sciences. He is a

[Figure 21]

recipient of the distinguished paper award in AAAI 2021. His group achieved the first runner-up at the ImageNet Large Scale Visual Recognition Challenge 2015 in scene recognition, and the winner at the ActivityNet Large Scale Activity Recognition Challenge 2016 in video classification. He served as the program chair of IEEE ICIST 2014.

Kaipeng Zhang received a B.S. degree in computer science from Donghua University, Shanghai, China, in 2016, an M.S. degree in computer science and information engineering from National Taiwan University, Taipei, Taiwan, in 2018, and a Ph.D. degree in Information Science from the University of Tokyo, Tokyo, Japan. He is currently a researcher at the Shanghai AI Laboratory. His research interests include face analysis, metric learning, and foundation models. He is a program committee member or reviewer for major international conferences and

[Figure 22]

journals, such as CVPR, ICCV, ECCV, NeurIPS, ICML, and TPAMI. He is also a senior program committee member for IJCAI 2021.

Ping Luo is an Associate Professor in the Department of Computer Science, The University of Hong Kong (HKU). He received his PhD degree in 2014 in Information Engineering, from the Chinese University of Hong Kong (CUHK), supervised by Prof. Xiaoou Tang and Prof. Xiaogang Wang. He was a Postdoctoral Fellow at CUHK from 2014 to 2016. He joined SenseTime Research as a Principal Research Scientist from 2017 to 2018. His research interests are machine learning and computer vision. He has published 100+ peer-reviewed articles in top-tier conferences and

[Figure 23]

journals such as TPAMI, IJCV, ICML, ICLR, CVPR, and NIPS.

[Figure 24]

Fanqing Meng received the B.S. degree in the School of Software Engineering, Tongji University, Shanghai, China. He is a first-year Ph.D. student in the School of Electronic Information and Electrical Engineering, at Shanghai Jiao Tong University, Shanghai, China. His current research interest focuses on the applications of computer vision as well as multimodal and transfer learning.

