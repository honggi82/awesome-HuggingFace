## VLMEvalKit: An Open-Source Toolkit for Evaluating Large Multi-Modality Models

# arXiv:2407.11691v4[cs.CV]28Aug2025

Haodong Duan Xinyu Fang Junming Yang Xiangyu Zhao Yuxuan Qiao

Mo Li Amit Agarwal Zhe Chen Lin Chen Yuan Liu Yubo Ma Hailong Sun Yifan Zhang Shiyin Lu Tack Hwa Wong Weiyun Wang Peiheng Zhou Xiaozhe Li

Chaoyou Fu Junbo Cui Jixuan Chen Enxin Song Song Mao Shengyuan Ding Tianhao Liang Zicheng Zhang Xiaoyi Dong Yuhang Zang Pan Zhang Jiaqi Wang

Dahua Lin Kai Chen 1VLMEvalKit Team 2Community Contributors

### Abstract

We present VLMEvalKit: an open-source toolkit for evaluating large multi-modality models based on PyTorch. The toolkit aims to provide a user-friendly and comprehensive framework for researchers and developers to evaluate existing multi-modality models and publish reproducible evaluation results. In VLMEvalKit, we implement over 200+ different large multi-modality models, including both proprietary APIs and open-source models, as well as more than 80 different multi-modal benchmarks. By implementing a single interface, new models can be easily added to the toolkit, while the toolkit automatically handles the remaining workloads, including data preparation, distributed inference, prediction post-processing, and metric calculation. Although the toolkit is currently mainly used for evaluating large vision-language models, its design is compatible with future updates that incorporate additional modalities, such as audio and video. Based on the evaluation results obtained with the toolkit, we host OpenVLM Leaderboard, a comprehensive leaderboard to track the progress of multi-modality learning research. The toolkit is released on GitHub and is actively maintained1.

### 1. Introduction

With the rapid development of Large Language Models (LLMs) [6,99,123,127], Large Multi-Modality Models (LMMs) [100, 121] have also experienced significant ad-

1VLMEvalKit contributors can join the author list of the report based on their contribution to the repository. Specifically, it requires 3 major contributions (implement a new benchmark, MLLM, or contribute a major feature). We will update the report quarterly and an additional section that details each developer’s contribution will be appended in the next update.

vancements. LMMs typically take two or more modalities as input. Most of the research has focused on LMMs for image and text [12,79], but research has also been extended to other modalities, such as audiotext [23], video [14,68,113], or point clouds [156,159]. Furthermore, there exist LMMs that can simultaneously take more than two modalities as inputs, including proprietary APIs and open-source models [44, 100, 121]. Compared to previous multi-modality models, LMMs, empowered by large language models, exhibit enhanced generalization capability and engage with humans in a variety of conversational styles. These models have not only demonstrated remarkable capabilities in multi-modal perception and reasoning tasks but have also spurred a range of innovative applications.

Quantitative evaluation is crucial in the development of LMMs. As general-purpose models, LMMs must undergo rigorous evaluation in a diverse range of tasks and domains. Comprehensive evaluations not only help users discern the strengths and weaknesses of an LMM, but also offer valuable feedback to developers for ongoing refinement. Unlike ‘pre-GPT’ models, evaluating LMMs on diversified quantitative benchmarks has become a common practice, both in academic works [78,165] and commercial APIs [100,121].

Despite the importance of evaluating LMMs, conducting assessments across dozens of benchmarks can be a daunting task, particularly for small research teams. One must prepare data based on numerous repositories and manage potential environmental conflicts. Moreover, the authors of benchmarks may not provide evaluation results for all LMMs in which users are interested, thus requiring significant effort to compile uncompleted results. To alleviate this challenge, we developed VLMEvalKit, an open-source toolkit designed to facilitate the evaluation of LMMs.

VLMEvalKit aims to provide a comprehensive, user-

|Benchmark Keyword<br><br>|Benchmark Keyword|
|---|---|
|MMBench [80] All-round Capability MME [33] All-round Capability<br><br>SEEDBench2-Plus [62] Text-Rich Comprehension MMMU [170] Multi-Modal Examination MathVista [87] Multi-Modal Math<br><br>OCRBench [82] Text Understanding AI2D [58] Diagram Understanding<br><br>DocVQA [97] Document Understanding HallusionBench [77] Hallucination & Illusion<br><br>MMBench-Video [31] Video Understanding MLLMGuard [41] Multi-Modal Safety MMVet [168] Subjective Evaluation|CCBench Chinese Culture<br><br>SEEDBench [64] All-round Capability<br><br>MMT-Bench [167] Multi-Task<br><br>MMStar [13] Data Contamination<br><br>COCO Caption [16] Image Caption<br><br>TextVQA [112] Text Recognition<br><br>ChartQA [95] Chart Understanding<br><br>RealWorldQA Driving Scenarios<br><br>POPE [70] Object Hallucination<br><br>OCRVQA [98] Text Recognition<br><br>AesBench [53] Aesthetics<br><br>LLaVABench [79] Subjective Evaluation|

[Figure 1]

Figure 1. VLMEvalKit supports most mainstream commercial APIs including GPT-4o, Gemini, GLM-4v, etc., as well as over 200 open-source LMMs including LLaVA [79], InternVL [19], Idefics [61], etc.

Table 1. VLMEvalKit supports the evaluation of more than 80 different benchmarks, covering diversified tasks, topics, and scenarios. In the table, we list some benchmarks and the corresponding keywords.

friendly evaluation framework for researchers and developers to assess existing Large Multi-Modality Models. Currently, the codebase supports more than 200 different large multi-modality models, spaning proprietary APIs and opensource models (see Fig. 1), and over 80 multi-modal benchmarks covering a wide range of tasks and scenarios (see Tab. 1). The codebase’s straightforward design simplifies the integration of new benchmarks or LMMs. Typically, a developer only needs to prepare a single data file or implement a single interface to support new benchmarks or LMMs. Beyond its extensive collection and simplified design, VLMEvalKit significantly eases the work of comprehensive evaluation. Users can launch evaluations across multiple supported LMMs and benchmarks with a single command, generating well-structured evaluation results. The entire process eliminates the need for manual data preparation or post-processing, ensuring a seamless and efficient evaluation experience.

Benchmarks LMMs

Multi-Modal Generation Interface

Data Files Prompt Building Interface

Custom Prompt (optional)

Multi-modal Inference

Multi-process Inference for API models

Multi-process Inference for open-source models

Multi-modal Evaluation

LLM Augmented Answer Extraction

Circular Evaluation Dataset-specific Evaluations

Figure 2. Major Components in VLMEvalKit.

of VLMEvalKit, present and analyze the evaluation results obtained, and discuss potential future developments.

To align with real use cases, VLMEvalKit employs generation-based evaluation across all LMMs and benchmarks. Given their general-purpose nature, LMMs may struggle with adhering to specific formats for solving multiple-choice questions. For a fair comparison, VLMEvalKit utilizes large language models as choice extractors when exact matching fails, thereby mitigating the impact of response styles and enhancing the reliability of evaluation, Leveraging the reliable and reproducible evaluation results provided by VLMEvalKit, we maintain a comprehensive leaderboard to monitor the advancement of LMM development. The leaderboard disseminates valuable insights to the community and has garnered widespread recognition from both academia and industry.

### 2. VLMEvalKit: Design & Features

Fig. 2 displays the major components in VLMEvalKit. In this section, we will describe each single component in detail.

Benchmarks. VLMEvalKit encompasses dozens of multimodal benchmarks that span a broad spectrum of tasks and scenarios, as detailed in Tab. 1. The benchmarks are first processed into .tsv files, where evaluation samples are stored as individual lines. Each evaluation sample typically includes the index, question, answer, image (one or more images encoded in base64), and choices for multi-choice questions. Each dataset class supports a .build prompt() interface, which the evaluation script utilizes to construct evaluation samples into multi-modal messages. A multi-modal message comprises an interleaved sequence of contents across various modalities. For illustration, consider the following example:

VLMEvalKit is publicly available at https : / / github.com/open-compass/VLMEvalKit under the Apache 2.0 License. The repository includes the complete source codes along with detailed instructions for installation, evaluation, and further development. In subsequent sections, we will delve into the design and features

[

dict(type=’image’, value=’path or url of the image’),

dict(type=’image’, value=’path or url of the image’),

dict(type=’text’, value=’Please list all the objects that appear in the above images.’) ]

Currently, a multi-modal message primarily incorporates image, video and text modalities. The format, however, is extensible and can accommodate additional modalities such as audio or point clouds.

LMMs. VLMEvalKit supports over 200 LMMs, including both commercial APIs and open-source models. A unified .generate() interface has been implemented for all LMMs, which accepts a multi-modal message as input and returns the response string. For LMMs that are limited to processing a single image-text pair, the concatenated text messages and the first image are adopted as the default input. To provide flexibility for LMM developers, the .generate() interface also includes dataset name as an additional argument. An LMM can optionally implement its own .build prompt() interface to construct custom multi-modal messages, and determine whether to utilize custom multi-modal messages based on the dataset name flag. The unified interface eases the process of comprehensive evaluation. Each newly supported LMM / benchmark can be directly evaluated with all existing benchmarks / LMMs.

Multi-modal Inference. To expedite the multi-modal inference, we support parallelized inference for both commercial APIs, leveraging Python’s multiprocessing, and open-source models, which are distributed in parallel across multiple GPUs. Additionally, we have implemented a robust inference process so that an interrupted inference process can be resumed with minimal costs of repeated calculation or API calls. For some popular open-source LMMs(like Qwen Series, InternVL Series), we have supported vllm/LMDeploy to accelerate the inference with full utilization of resources.

Multi-modal Evaluation. Predictions from the LMMs will be evaluated based on the specific question format to derive the final metrics. Benchmarks supported in VLMEvalKit can be categorized into three primary types: 1. Multi-choice questions (MCQ), where the model must select from given options and respond with the corresponding label (e.g., A, B); 2. Yes-or-No questions (Y/N), requiring a straightforward ‘Yes’ or ‘No’ answer; 3. Open-ended questions, which necessitate a free-form response. Notably, a significant number of LMMs struggle to adhere to the instructions precisely, often producing responses that are not well-formatted for MCQ and Y/N benchmarks. To improve

the precision of evaluations and counteract the influence of varied response styles, VLMEvalKit offers the option to integrate LLM-augmented answer extraction specifically for MCQ and Y/N benchmarks. We first adopt exact matching to match the response with the option labels or contents. Should this step fail, the toolkit then prompts an LLM (such as ChatGPT) to match the response with the option that most closely aligns with semantic meaning. This strategy helps us better understand the real performance of LMMs, particularly for commercial APIs [80].

Another notable challenge in assessing MCQ benchmarks is the inherent variance. When employing random guessing, an LMM may correctly answer 1/N questions for N-option multi-choice benchmarks. Besides, we find that the outcome of an LMM can be significantly influenced by the order of options. These factors make the evaluation results highly variable and the performance gap between LMMs less discernible. To address this, VLMEvalKit offers an option to evaluate all MCQ benchmarks in Circular mode: N options of a MCQ will be shifted in circular N times to formulate N new questions. The results count only if an LMM accurately answers all N circular-shifted MCQs. The CircularEval strategy can more effectively assess the real comprehension of an LMM on MCQ benchmarks, allowing users to identify more pronounced performance disparities between models.

For open-ended benchmarks, VLMEvalKit follows the original practice for conducting the evaluation. Subjective benchmarks like MMVet [168] or LLaVABench [79] adopts GPT-4 for marking, based on the semantic similarity between LMM responses and the reference answer. For VQA benchmarks [98, 112], we follow the standard practices to calculate accuracies based on heuristic matching.

### 3. Evaluation Results

#### 3.1. General VQA Benchmarks

Utilizing VLMEvalKit, one can conduct comprehensive evaluations of an LMM across numerous benchmarks to gain a thorough understanding of its strengths and weaknesses. We publish all evaluation results on OpenVLM Leaderboard. Our core leaderboard is based on evaluations from eight distinct benchmarks: 1. MMBench v1.1 [test] [80]2 (all-round capability); 2. MMStar [13] (data contamination); 3. MMMU [170] [val] (multi-modal examination); 4. MathVista [mini-test] [87] (multi-modal math); 5. HallusionBench [77] (hallucination & illusion); 6. AI2D [test] [58] (diagram understanding); 7. OCRBench [82] (text understanding); 8. MMVet [168] (subjective evaluation). The selection encompasses a diverse array of tasks, and the average score across these benchmarks serves as a reliable indicator of the general capabilities of LMMs. In

2We report the average score of English and Chinese test splits.

Method Param. Avg. Score MMBenchV1.1 MMStar MMMU Math. Hallu. AI2D OCR. MMVet CongRong-v2.0 [24] N/A 80.7 88.1 75.3 75.6 76.8 63.2 90.0 927 83.9

SenseNova-V6-Pro [108] N/A 80.4 88.0 73.7 70.4 76.9 67.1 89.2 895 88.2 Gemini-2.5-Pro [121] N/A 80.1 88.3 73.6 74.7 80.9 64.1 89.5 862 83.3 InternVL3-78B [189] 78B 79.1 87.7 73.4 72.2 79.0 59.1 89.8 908 80.7 InternVL3-38B [189] 38B 77.8 86.8 72.6 69.7 76.3 58.4 88.7 886 81.1

Step-1o [115] N/A 77.7 87.3 69.3 69.9 74.7 55.8 89.1 926 82.8

SenseNova [108] N/A 77.4 85.7 72.7 69.6 78.4 57.4 87.8 894 78.2 InternVL2.5-78B-MPO [18] 78B 77 87.7 72.1 68.2 76.6 58.1 89.2 909 73.5 GLM-4v-plus-20250111 [5] N/A 76.7 85.9 72.5 69.9 73.5 58.5 86.7 908 75.7

Ovis2-34B [89] 34B 76.5 86.5 69.2 66.7 76.1 58.8 88.3 894 77.7

- Table 2. The evaluation results of LMMs on general VQA benchmarks. The table displays the top-10 LMMs, including both commercial APIs and open-source LMMs(till 2025.06.20), in the descending order of average score. When calculating the average score, scores of each benchmark are normalized to the range of 0 to 100. When reporting the parameter size, ‘1B’ means 109 parameters. Commercial APIs are denoted with blue background . Bold and underline indicates the best and second-best performance in each group.

Method Param. Avg. Score MathVista MathVision MathVerse DynaMath WeMath LogicVista

Seed1.5-VL [42] N/A 73.3 86.8 67.3 79.3 56.1 77.5 72.7 Gemini-2.5-Pro [121] N/A 72.5 80.9 69.1 76.9 56.3 78 73.8 Doubao-1.5-pro [120] N/A 61.6 78.6 51.5 64.7 44.9 65.7 64.2 Gemini-2.0-Pro [121] N/A 56.6 71.3 48.1 67.3 43.3 56.5 53.2

ChatGPT-4o-latest [99] N/A 54.8 71.6 43.8 49.9 48.5 50.6 64.4 GPT-4.1-20250414 [99] N/A 54 70.4 45.1 48.9 43.3 55.5 61.1

InternVL3-78B [99] 78B 51 79 38.8 51 35.1 46.1 55.9 Gemini-2.0-flash [121] N/A 50.6 70.4 43.6 47.8 42.1 47.4 52.3

Claude-3.7-sonnet-20250219 [6] N/A 50.4 66.8 41.9 46.7 39.7 49.3 58.2 InternVL3-38B [189] 38B 50.3 76.3 35.9 48.2 34.7 49.3 57.7

- Table 3. The evaluation results of LMMs on image reasoning benchmarks. The table displays the top-10 LMMs, including both commercial APIs and open-source LMMs(till 2025.02.28), in the descending order of average score.

Tab. 2, we present the performance of top-10 commercial APIs and the top-10 open-source models evaluated on this suite of benchmarks.

As illustrated in the table, after nearly two years of development, the performance gap between commercial APIs and open-source LMMs on general VQA benchmarks has significantly narrowed. Open-source LMMs now demonstrate strong capabilities in general understanding tasks, often matching or even surpassing the performance of commercial APIs. Among the top-10 models, four are commercial APIs, while the other five are open-source models. Although the top three models, CongRong-v2.0 [24], SenseNova-V6-Pro [108], and Gemini-2.5-Pro [121] remain commercial APIs, their performance is remarkably close to that of the fourth-ranked model, InternVL378B [189]. From the first-ranked model, CongRong-v2.0, to the tenth-ranked model, Ovis2-34B, the average score

decreases by only 4.2% (from 80.7% to 76.5%). Notably, prominent commercial series such as GPT [99] and Claude [6] fail to rank within the top 10, whereas opensource models like the InternVL and Ovis series exhibit exceptional performance and competitive strength.

#### 3.2. Image Reasoning Benchmarks

With the rapid advancement of LMMs, reasoning capabilities have garnered increasing attention. To systematically assess these capabilities, we have compiled several widely recognized multi-modal benchmarks designed for reasoning tasks, including MathVista [87], MathVision [136], MathVerse [176], DynaMath [190], WeMath [105], and LogicVista [154]. Evaluations were conducted on OpenVLM Reasoning Leaderboard. The selected benchmarks primarily emphasize mathematical and puzzlesolving problems, spanning areas such as geometry, alge-

Method Param. Frame. Avg. Score MVBench Video-MME(w/o subs) MMBench-Video TempCompass MLVU

InternVL3-78B [189] 78.4 64 72.7 79.2 73.1 1.81 77.0 74.0 InternVL3-38B [189] 38.4 64 71.9 76.0 72.1 1.80 78.5 72.7 Qwen2.5-VL-72B [7] 73.4 64 68.6 71.3 68.6 1.84 77.7 64.1 InternVL2.5-78B [18] 78.4 64 68.1 75.6 72.1 1.98 56.0 70.9 Qwen2-VL-72B [137] 73.4 64 67.7 70.2 67.3 1.70 79.4 65.2

InternVL3-8B [189] 7.94 64 66.8 73.2 66.0 1.69 70.4 68.0 LLaVA-Video-72B-Qwen2 [179] 7.94 64 66.2 63.1 70.5 1.71 70.4 70.0

Aria [66] 25.3 64 66.2 67.9 66.0 1.81 69.6 67.1 Gemini-2.0-flash [121] N/A 64 64.6 59.4 71.0 2.01 57.9 68.0 GPT-4o-20240806 [99] N/A 16 63.1 57.5 67.9 1.87 72.7 54.9

Qwen2.5-Omni-7B [155] 10.7 64 64.5 69.0 64.1 1.65 70.7 63.6

- Table 4. The evaluation results of LMMs on video understanding benchmarks. The table displays the top-10 LMMs, including both commercial APIs and open-source LMMs(till 2025.06.25), in the descending order of average score. Due to resource limitations, we only tested a few commercial APIs.

bra, logic, and related domains. The performance metrics of the top-10 models are summarized in Tab. 3.

In multi-modal reasoning tasks, a significant performance gap remains evident between commercial APIs and open-source LMMs. Generally, commercial APIs demonstrate a marked advantage over their open-source counterparts. Notably, the top six performing LMMs are all proprietary APIs. InternVL3-78B [189], the highest-performing open-source LMM on reasoning benchmarks, achieves an average score of only 51, highlighting a considerable gap compared to the leading model Seed1.5-VL [42](73.3). Further analysis reveals that most high-performing models achieve accuracy exceeding 85% on MathVista [87]. In contrast, only Seed1.5-VL and Gemini-2.5-Pro achieve accuracy greater than 50% on DynaMath [190], underscoring the challenges posed by this benchmark.

#### 3.3. Video Benchmarks

Video has become essential in daily life, driving communication, learning, and entertainment. With the rapid rise of video content and progress in LMM capability, video understanding ability is increasingly vital for LMMs. To fully examine the video understanding capability, we have added support for some popular common video understanding benchmarks, including MMBenchVideo [31], Video-MME [34], MLVU [188], TempCompass [81], MVBench [68], and other video datasets. Besides, in order to have a glimpse into the overall video understanding performance of LMMs, we also selected some representative video benchmarks and evaluated over forty LMMs, with the results presented on the OpenVLM Video Leaderboard. The selected benchmarks primarily cover various question types (multiple choice, open-ended questions, true/false questions, etc.) and various video lengths, mainly examining the model’s general video understanding ability.

The performance metrics of the top-10 models are summarized in Tab. 4.

Due to the limited resources of commercial APIs and the significant costs associated with video understanding evaluations, we focused on testing the video understanding capabilities of open-source LMMs. It’s easy to observe that InternVL and Qwen Series demonstrate remarkable performance in video understanding benchmarks. InternVL378B [189] achieves the highest average score of 72.7, and Qwen2.5 lags slightly behind them, mainly revealed in Video-MME [34] and MLVU [188]. With a 64-frame input window, Gemini-2.0-flash [121] still lags behind the state of the art on several tasks, possibly because it refuses to answer some questions. GPT-4o [99] achieves 63.1 with only 16 frames, showing that commercial APIs can still compete in video understanding even under a short context.

### 4. Discussion

We have released VLMEvalKit, an open-source toolkit designed for the evaluation of large multi-modality models. VLMEvalKit encompasses a comprehensive collection of over 300 LMMs and more than 100 multi-modal benchmarks. The codebase is structured with simplicity in mind, facilitating the easy integration of new LMMs or benchmarks. Our framework is thoughtfully designed to extend its capabilities beyond the vision modality and incorporate other modalities like audio. Moving forward, the development of VLMEvalKit will focus on expanding the repertoire of LMMs and benchmarks for video and other modalities. We are optimistic that this repository, along with all released resources, will contribute to advancing research in multi-modal learning.

### References

- [1] Marah Abdin, Jyoti Aneja, Harkirat Behl, S´ebastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024. 18
- [2] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, et al. Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219,

2024. 16

- [3] Manoj Acharya, Kushal Kafle, and Christopher Kanan. Tallyqa: Answering complex counting questions. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pages 8076–8084, 2019. 18
- [4] Amit Agarwal, Srikant Panda, Angeline Charles, Bhargava Kumar, Hitesh Patel, Priyaranjan Pattnayak, Taki Hasan Rafi, Tejaswini Kumar, and Dong-Kyu Chae. Mvtamperbench: Evaluating robustness of vision-language models. arXiv preprint arXiv:2412.19794, 2024. 16
- [5] Zhipu AI. Glm-4v. https://open.bigmodel.cn/ dev/howuse/glm-4v, 2024. 4
- [6] Anthropic. The claude 3 model family: Opus, sonnet, haiku. 2024. 1, 4
- [7] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 5
- [8] Wenhao Chai, Enxin Song, Yilun Du, Chenlin Meng, Vashisht Madhavan, Omer Bar-Tal, Jenq-Neng Hwang, Saining Xie, and Christopher D Manning. Auroracap: Efficient, performant video detailed captioning and a new benchmark. arXiv preprint arXiv:2410.03051, 2024. 17
- [9] Guo Chen, Yicheng Liu, Yifei Huang, Yuping He, Baoqi Pei, Jilan Xu, Yali Wang, Tong Lu, and Limin Wang. Cg-bench: Clue-grounded question answering benchmark for long video understanding. arXiv preprint arXiv:2412.12075, 2024. 18
- [10] Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. Sft or rl? an early investigation into training r1-like reasoning large vision-language models. arXiv preprint arXiv:2504.11468,

2025. 18

- [11] Jiacheng Chen, Tianhao Liang, Sherman Siu, Zhengqing Wang, Kai Wang, Yubo Wang, Yuansheng Ni, Wang Zhu, Ziyan Jiang, Bohan Lyu, et al. Mega-bench: Scaling multimodal evaluation to over 500 real-world tasks. arXiv preprint arXiv:2410.10563, 2024. 17
- [12] Lin Chen, Jisong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv:2311.12793, 2023. 1, 17

- [13] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, et al. Are we on the right way for evaluating large vision-language models? arXiv:2403.20330, 2024. 2, 3, 16
- [14] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and generation with better captions. arXiv preprint arXiv:2406.04325, 2024. 1
- [15] Pengcheng Chen, Jin Ye, Guoan Wang, Yanjun Li, Zhongying Deng, Wei Li, Tianbin Li, Haodong Duan, Ziyan Huang, Yanzhou Su, et al. Gmai-mmbench: A comprehensive multimodal evaluation benchmark towards general medical ai. arXiv preprint arXiv:2408.03361, 2024. 16, 17
- [16] Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco captions: Data collection and evaluation server. arXiv:1504.00325, 2015. 2
- [17] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811,

2025. 18

- [18] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024. 4, 5
- [19] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, Ji Ma, Jiaqi Wang, Xiaoyi Dong, Hang Yan, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites,

2024. 2, 16

- [20] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv:2312.14238, 2023. 16
- [21] Junhao Cheng, Yuying Ge, Teng Wang, Yixiao Ge, Jing Liao, and Ying Shan. Video-holmes: Can mllm think like holmes for complex video reasoning? arXiv preprint arXiv:2505.21374, 2025. 18
- [22] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935, 2024. 17
- [23] Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, et al. Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models. arXiv:2311.07919, 2023. 1
- [24] CloudWalk. Congrong-v2.0. https : / / mllm . cloudwalk.com/web/, 2025. 4
- [25] XTuner Contributors. Xtuner: A toolkit for efficiently fine-tuning llm. https://github.com/InternLM/ xtuner, 2023. 17

- [26] WeChat CV. Wemm. https://github.com/ scenarios/WeMM, 2023. 17
- [27] Shengyuan Ding, Shenxi Wu, Xiangyu Zhao, Yuhang Zang, Haodong Duan, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Mm-ifengine: Towards multimodal instruction following. arXiv preprint arXiv:2504.07957, 2025. 17
- [28] Khang T. Doan, Bao G. Huynh, Dung T. Hoang, Thuc D. Pham, Nhat H. Pham, Quan T. M. Nguyen, Bang Q. Vo, and Suong N. Hoang. Vintern-1b: An efficient multimodal large language model for vietnamese, 2024. 18
- [29] Hongyuan Dong, Zijian Kang, Weijie Yin, Xiao Liang, Chao Feng, and Jiao Ran. Scalable vision language model training via high quality data curation. arXiv preprint arXiv:2501.05952, 2025. 18
- [30] Xinyu Fang, Zhijian Chen, Kai Lan, Lixin Ma, Shengyuan Ding, Yingji Liang, Xiangyu Zhao, Farong Wen, Zicheng Zhang, Guofeng Zhang, et al. Creation-mmbench: Assessing context-aware creative intelligence in mllm. arXiv preprint arXiv:2503.14478, 2025. 17
- [31] Xinyu Fang, Kangrui Mao, Haodong Duan, Xiangyu Zhao, Yining Li, Dahua Lin, and Kai Chen. Mmbench-video: A long-form multi-shot benchmark for holistic video understanding. arXiv:2406.14515, 2024. 2, 5, 16
- [32] Kaiyue Feng, Yilun Zhao, Yixin Liu, Tianyu Yang, Chen Zhao, John Sous, and Arman Cohan. Physics: Benchmarking foundation models on university-level physics problem solving. arXiv preprint arXiv:2503.21821, 2025. 18
- [33] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. ArXiv, abs/2306.13394,

2023. 2, 14

- [34] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 5, 16
- [35] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 17
- [36] Chaoyou Fu, Haojia Lin, Zuwei Long, Yunhang Shen, Meng Zhao, Yifan Zhang, Shaoqi Dong, Xiong Wang, Di Yin, Long Ma, et al. Vita: Towards open-source interactive omni multimodal llm. arXiv preprint arXiv:2408.05211,

2024. 17

- [37] Chaoyou Fu, Haojia Lin, Xiong Wang, Yi-Fan Zhang, Yunhang Shen, Xiaoyu Liu, Yangze Li, Zuwei Long, Heting Gao, Ke Li, et al. Vita-1.5: Towards gpt-4o level real-time vision and speech interaction. arXiv preprint arXiv:2501.01957, 2025. 17
- [38] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu

- Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv:2404.12390, 2024. 15, 17
- [39] Zhangwei Gao, Zhe Chen, Erfei Cui, Yiming Ren, Weiyun Wang, Jinguo Zhu, Hao Tian, Shenglong Ye, Junjun He, Xizhou Zhu, et al. Mini-internvl: a flexible-transfer pocket multi-modal model with 5% parameters and 90% performance. Visual Intelligence, 2(1):1–17, 2024. 16
- [40] Shuhao Gu, Jialing Zhang, Siyuan Zhou, Kevin Yu, Zhaohu Xing, Liangdong Wang, Zhou Cao, Jintao Jia, Zhuoyi Zhang, Yixuan Wang, Zhenchong Hu, Bo-Wen Zhang, Jijie Li, Dong Liang, Yingli Zhao, Yulong Ao, Yaoqi Liu, Fangxiang Feng, and Guang Liu. Infinity-mm: Scaling multimodal performance with large-scale and high-quality instruction data, 2024. 18
- [41] Tianle Gu, Zeyang Zhou, Kexin Huang, Dandan Liang, Yixu Wang, Haiquan Zhao, Yuanqi Yao, Xingge Qiao, Keqing Wang, Yujiu Yang, et al. Mllmguard: A multidimensional safety evaluation suite for multimodal large language models. arXiv preprint arXiv:2406.07594, 2024. 2, 16
- [42] Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025. 4, 5
- [43] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617,

2018. 17

- [44] Jiaming Han, Kaixiong Gong, Yiyuan Zhang, Jiaqi Wang, Kaipeng Zhang, Dahua Lin, Yu Qiao, Peng Gao, and Xiangyu Yue. Onellm: One framework to align all modalities with language. In CVPR, pages 26584–26595, 2024. 1
- [45] Xiaotian Han, Quanzeng You, Yongfei Liu, Wentao Chen, Huangjie Zheng, Khalil Mrini, Xudong Lin, Yiqi Wang, Bohan Zhai, Jianbo Yuan, et al. Infimm-eval: Complex open-ended reasoning evaluation for multi-modal large language models. arXiv e-prints, pages arXiv–2311, 2023. 17
- [46] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024. 17
- [47] Muyang He, Yexin Liu, Boya Wu, Jianhao Yuan, Yueze Wang, Tiejun Huang, and Bo Zhao. Efficient multimodal learning from data-centric perspective. arXiv preprint arXiv:2402.11530, 2024. 17
- [48] Jack Hong, Shilin Yan, Jiayin Cai, Xiaolong Jiang, Yao Hu, and Weidi Xie. Worldsense: Evaluating real-world omnimodal understanding for multimodal llms. arXiv preprint arXiv:2502.04326, 2025. 18
- [49] Kaiyuan Hou, Minghui Zhao, Lilin Xu, Yuang Fan, and Xiaofan Jiang. Tdbench: Benchmarking vision-language models in understanding top-down images. arXiv preprint arXiv:2504.03748, 2025. 18

- [50] Hailang Huang, Yong Wang, Zixuan Huang, Huaqiu Li, Tongwen Huang, Xiangxiang Chu, and Richong Zhang. Mmgenbench: Evaluating the limits of lmms from the text-to-image generation perspective. arXiv preprint arXiv:2411.14062, 2024. 18
- [51] Mingxin Huang, Yuliang Liu, Dingkang Liang, Lianwen Jin, and Xiang Bai. Mini-monkey: Multi-scale adaptive cropping for multimodal large language models. arXiv preprint arXiv:2408.02034, 2024. 18
- [52] Mingxin Huang, Yongxin Shi, Dezhi Peng, Songxuan Lai, Zecheng Xie, and Lianwen Jin. Ocr-reasoning benchmark: Unveiling the true capabilities of mllms in complex textrich image reasoning. arXiv preprint arXiv:2505.17163,

2025. 18

- [53] Yipo Huang, Quan Yuan, Xiangfei Sheng, Zhichao Yang, Haoning Wu, Pengfei Chen, Yuzhe Yang, Leida Li, and Weisi Lin. Aesbench: An expert benchmark for multimodal large language models on image aesthetics perception. arXiv preprint arXiv:2401.08276, 2024. 2, 15, 17
- [54] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 16
- [55] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max W.F. Ku, Qian Liu, and Wenhu Chen. Mantis: Interleaved multiimage instruction tuning. Transactions on Machine Learning Research, 2024, 2024. 17
- [56] Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanwei Li, Yu Qi, Xinyan Chen, Liuhui Wang, Jianhan Jin, Claire Guo, Shen Yan, et al. Mme-cot: Benchmarking chain-of-thought in large multimodal models for reasoning quality, robustness, and efficiency. arXiv preprint arXiv:2502.09621, 2025. 18
- [57] Peng Jin, Ryuichi Takanobu, Wancai Zhang, Xiaochun Cao, and Li Yuan. Chat-univi: Unified visual representation empowers large language models with image and video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13700– 13710, 2024. 16
- [58] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In ECCV, pages 235–251. Springer,

2016. 2, 3

- [59] Yoonsik Kim, Moonbin Yim, and Ka Yeon Song. Tablevqabench: A visual question answering benchmark on multiple table domains. arXiv preprint arXiv:2404.19205, 2024. 14, 17
- [60] Hugo Lauren¸con, Andr´es Marafioti, Victor Sanh, and L´eo Tronchon. Building and better understanding visionlanguage models: insights and future directions. In Workshop on Responsibly Building the Next Generation of Multimodal Foundational Models, 2024. 16
- [61] Hugo Laurenc¸on, Lucile Saulnier, L´eo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, et al. Obelics: An open web-scale filtered dataset of interleaved image-text documents, 2023. 2

- [62] Bohao Li, Yuying Ge, Yi Chen, Yixiao Ge, Ruimao Zhang, and Ying Shan. Seed-bench-2-plus: Benchmarking multimodal large language models with text-rich visual comprehension. arXiv preprint arXiv:2404.16790, 2024. 2, 14
- [63] Baiqi Li, Zhiqiu Lin, Wenxuan Peng, Jean de Dieu Nyandwi, Daniel Jiang, Zixian Ma, Simran Khanuja, Ranjay Krishna, Graham Neubig, and Deva Ramanan. Naturalbench: Evaluating vision-language models on natural adversarial samples. Advances in Neural Information Processing Systems, 37:17044–17068, 2025. 18
- [64] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv:2307.16125, 2023. 2, 14
- [65] Chunyi Li, Jianbo Zhang, Zicheng Zhang, Haoning Wu, Yuan Tian, Wei Sun, Guo Lu, Xiaohong Liu, Xiongkuo Min, Weisi Lin, et al. R-bench: Are your large multimodal model robust to real-world corruptions? arXiv preprint arXiv:2410.05474, 2024. 18
- [66] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Guoyin Wang, Bei Chen, and Junnan Li. Aria: An open multimodal native mixtureof-experts model. arXiv preprint arXiv:2410.05993, 2024. 5, 18
- [67] Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: Gui grounding for professional highresolution computer use. arXiv preprint arXiv:2504.07981,

2025. 17

- [68] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mvbench: A comprehensive multi-modal video understanding benchmark. In CVPR, pages 22195–22206, 2024. 1, 5, 16
- [69] Lei Li, Yuancheng Wei, Zhihui Xie, Xuqing Yang, Yifan Song, Peiyi Wang, Chenxin An, Tianyu Liu, Sujian Li, Bill Yuchen Lin, et al. Vlrewardbench: A challenging benchmark for vision-language generative reward models. arXiv preprint arXiv:2411.17451, 2024. 18
- [70] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv:2305.10355, 2023. 2, 15
- [71] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pages 323–340. Springer, 2024. 16
- [72] Yanwei Li, Yuechen Zhang, Chengyao Wang, Zhisheng Zhong, Yixin Chen, Ruihang Chu, Shaoteng Liu, and Jiaya Jia. Mini-gemini: Mining the potential of multi-modality vision language models. arXiv preprint arXiv:2403.18814,

2024. 17

- [73] Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. Monkey: Image resolution and text label are important things for large multi-modal models. arXiv:2311.06607, 2023. 17
- [74] Yuan-Hong Liao, Rafid Mahmood, Sanja Fidler, and David Acuna. Reasoning paths with reference objects elicit quan-

- titative spatial reasoning in large vision-language models. arXiv preprint arXiv:2409.09788, 2024. 18
- [75] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26689–26699, 2024. 16
- [76] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014. 16
- [77] Fuxiao Liu, Tianrui Guan, Zongxia Li, Lichang Chen, Yaser Yacoob, et al. Hallusionbench: You see what you think? or you think what you see? an image-context reasoning benchmark challenging for gpt-4v (ision), llava-1.5, and other multi-modality models. arXiv:2310.14566, 2023. 2, 3, 15
- [78] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.

- 1

[79] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv:2304.08485, 2023. 1,

- 2, 3, 14

- [80] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, et al. Mmbench: Is your multi-modal model an all-around player? arXiv:2307.06281, 2023. 2, 3, 14
- [81] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? arXiv preprint arXiv:2403.00476, 2024. 5, 16
- [82] Yuliang Liu, Zhang Li, Hongliang Li, Wenwen Yu, Mingxin Huang, Dezhi Peng, Mingyu Liu, Mingrui Chen, Chunyuan Li, Lianwen Jin, et al. On the hidden mystery of ocr in large multimodal models. arXiv:2305.07895, 2023. 2, 3, 14, 17
- [83] Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, et al. Mmdu: A multi-turn multi-image dialog understanding benchmark and instruction-tuning dataset for lvlms. arXiv preprint arXiv:2406.11833, 2024. 15
- [84] Zuyan Liu, Yuhao Dong, Jiahui Wang, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao. Ola: Pushing the frontiers of omni-modal language model with progressive modality alignment. arXiv preprint arXiv:2502.04328,

2025. 18

- [85] DataCanvas Ltd. mmalaya. https://github.com/ DataCanvasIO/MMAlaya, 2024. 17
- [86] Lidong Lu, Guo Chen, Zhiqi Li, Yicheng Liu, and Tong Lu. Av-reasoner: Improving and benchmarking cluegrounded audio-visual counting for mllms. arXiv preprint arXiv:2506.05328, 2025. 18
- [87] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang,

- Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv:2310.02255, 2023. 2, 3, 4, 5, 15
- [88] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, KaiWei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022. 15
- [89] Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Han-Jia Ye. Ovis: Structural embedding alignment for multimodal large language model. arXiv:2405.20797, 2024. 4, 17
- [90] Ruilin Luo, Zhuofan Zheng, Yifan Wang, Yiyao Yu, Xinzhe Ni, Zicheng Lin, Jin Zeng, and Yujiu Yang. Ursa: Understanding and verifying chain-of-thought reasoning in multimodal mathematics. arXiv preprint arXiv:2501.04686,

2025. 18

- [91] Wufei Ma, Haoyu Chen, Guofeng Zhang, Celso M de Melo, Alan Yuille, and Jieneng Chen. 3dsrbench: A comprehensive 3d spatial reasoning benchmark. arXiv preprint arXiv:2412.07825, 2024. 18
- [92] Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, et al. Mmlongbench-doc: Benchmarking longcontext document understanding with visualizations. arXiv preprint arXiv:2407.01523, 2024. 15, 16
- [93] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023. 16
- [94] Damiano Marsili, Rohun Agrawal, Yisong Yue, and Georgia Gkioxari. Visual agentic ai for spatial reasoning with a dynamic api. arXiv preprint arXiv:2502.06787, 2025. 18
- [95] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv:2203.10244, 2022. 2, 14
- [96] Minesh Mathew, Viraj Bagal, Rub`en Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022. 14
- [97] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021. 2, 14
- [98] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. Ocr-vqa: Visual question answering by reading text in images. In ICDAR, pages 947–952. IEEE,

2019. 2, 3, 14

- [99] OpenAI. Chatgpt. https://openai.com/blog/ chatgpt, 2023. 1, 4, 5
- [100] OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774,

2023. 1

- [101] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv:2306.14824, 2023. 17

- [102] Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, et al. Humanity’s last exam. arXiv preprint arXiv:2501.14249, 2025. 17
- [103] Yukun Qi, Yiming Zhao, Yu Zeng, Xikun Bao, Wenxuan Huang, Lin Chen, Zehui Chen, Jie Zhao, Zhongang Qi, and Feng Zhao. Vcr-bench: A comprehensive evaluation framework for video chain-of-thought reasoning. arXiv preprint arXiv:2504.07956, 2025. 18
- [104] Yusu Qian, Hanrong Ye, Jean-Philippe Fauconnier, Peter Grasch, Yinfei Yang, and Zhe Gan. Mia-bench: Towards better instruction following evaluation of multimodal llms. arXiv preprint arXiv:2407.01509, 2024. 17
- [105] Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024. 4, 16
- [106] Yufan Ren, Konstantinos Tertikas, Shalini Maiti, Junlin Han, Tong Zhang, Sabine S¨usstrunk, and Filippos Kokkinos. Vgrp-bench: Visual grid reasoning puzzle benchmark for large vision-language models. arXiv preprint arXiv:2503.23064, 2025. 18
- [107] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In European conference on computer vision, pages 146–162. Springer, 2022. 16
- [108] Sensetime. Sensenova. https://sensenova.cn,

2024. 4

- [109] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025. 17
- [110] Hui Shen, Taiqiang Wu, Qi Han, Yunta Hsieh, Jizhou Wang, Yuyue Zhang, Yuxin Cheng, Zijian Hao, Yuansheng Ni, Xin Wang, et al. Phyx: Does your model have the” wits” for physical reasoning? arXiv preprint arXiv:2505.15929,

2025. 18

- [111] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, Bryan Catanzaro, Andrew Tao, Jan Kautz, Zhiding Yu, and Guilin Liu. Eagle: Exploring the design space for multimodal llms with mixture of encoders. arXiv:2408.15998, 2024. 17
- [112] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In CVPR, pages 8317–8326, 2019. 2, 3, 14
- [113] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, et al. Moviechat: From dense token to sparse memory for long video understanding. In CVPR, pages 18221– 18232, 2024. 1, 17
- [114] Enxin Song, Wenhao Chai, Weili Xu, Jianwen Xie, Yuxuan Liu, and Gaoang Wang. Video-mmlu: A massive multi-

- discipline lecture understanding benchmark. arXiv preprint arXiv:2504.14693, 2025. 17
- [115] StepFun. Step-1o. https://platform.stepfun. com, 2024. 4
- [116] Hai-Long Sun, Da-Wei Zhou, Yang Li, Shiyin Lu, Chao Yi, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, DeChuan Zhan, et al. Parrot: Multilingual visual instruction tuning. arXiv preprint arXiv:2406.02539, 2024. 15, 16
- [117] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. Slidevqa: A dataset for document visual question answering on multiple images. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 13636–13645, 2023. 15, 16
- [118] Jingqun Tang, Qi Liu, Yongjie Ye, Jinghui Lu, Shu Wei, Chunhui Lin, Wanqing Li, Mohamad Fitri Faiz Bin Mahmood, Hao Feng, Zhen Zhao, et al. Mtvqa: Benchmarking multilingual text-centric visual question answering. arXiv preprint arXiv:2405.11985, 2024. 15
- [119] Kexian Tang, Junyao Gao, Yanhong Zeng, Haodong Duan, Yanan Sun, Zhening Xing, Wenran Liu, Kaifeng Lyu, and Kai Chen. Lego-puzzles: How good are mllms at multi-step spatial reasoning? arXiv preprint arXiv:2503.19990, 2025. 18
- [120] Doubao Team. Doubao-1.5-pro. https://team. doubao.com, 2024. 4
- [121] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, et al. Gemini: a family of highly capable multimodal models. arXiv:2312.11805, 2023. 1, 4, 5
- [122] Granite Vision Team, Leonid Karlinsky, Assaf Arbelle, Abraham Daniels, Ahmed Nassar, Amit Alfassi, Bo Wu, Eli Schwartz, Dhiraj Joshi, Jovana Kondic, et al. Granite vision: a lightweight, open-source multimodal model for enterprise intelligence. arXiv preprint arXiv:2502.09927,

2025. 18

- [123] InternLM Team. Internlm: A multilingual language model with progressively enhanced capabilities. https://github.com/InternLM/InternLMtechreport, 2023. 1
- [124] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025. 17
- [125] Yang Tian, Zheng Lu, Mingqi Gao, Zheng Liu, and Bo Zhao. Mmcr: Benchmarking cross-source reasoning in scientific papers. arXiv preprint arXiv:2503.16856, 2025. 18
- [126] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024. 17
- [127] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv:2302.13971, 2023. 1, 17

- [128] Jordy Van Landeghem, Rub`en Tito, Łukasz Borchmann, Michał Pietruszka, Pawel Joziak, Rafal Powalski, Dawid Jurkiewicz, Micka¨el Coustaty, Bertrand Anckaert, Ernest Valveny, et al. Document understanding dataset and evaluation (dude). In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 19528– 19540, 2023. 15, 16
- [129] An-Lan Wang, Jingqun Tang, Liao Lei, Hao Feng, Qi Liu, Xiang Fei, Jinghui Lu, Han Wang, Weiwei Liu, Hao Liu, et al. Wilddoc: How far are we from achieving comprehensive and robust document understanding in the wild? arXiv preprint arXiv:2505.11015, 2025. 18
- [130] Fengxiang Wang, Mingshuo Chen, Xuming He, YiFan Zhang, Feng Liu, Zijie Guo, Zhenghao Hu, Jiong Wang, Jingyi Xu, Zhangrui Li, et al. Omniearth-bench: Towards holistic evaluation of earth’s six spheres and crossspheres interactions with multimodal observational earth data. arXiv preprint arXiv:2505.23522, 2025. 18
- [131] Fei Wang, Xingyu Fu, James Y Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, et al. Muirbench: A comprehensive benchmark for robust multi-image understanding. arXiv preprint arXiv:2406.09411, 2024. 15, 16
- [132] Fengxiang Wang, Hongzhen Wang, Zonghao Guo, Di Wang, Yulin Wang, Mingshuo Chen, Qiang Ma, Long Lan, Wenjing Yang, Jing Zhang, et al. Xlrs-bench: Could your multimodal llms understand extremely large ultrahigh-resolution remote sensing imagery? In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14325–14336, 2025. 18
- [133] Haochen Wang, Anlin Zheng, Yucheng Zhao, Tiancai Wang, Zheng Ge, Xiangyu Zhang, and Zhaoxiang Zhang. Reconstructive visual instruction tuning. arXiv preprint arXiv:2410.09575, 2024. 18
- [134] Junying Wang, Wenzhe Li, Yalun Wu, Yingji Liang, Yijin Guo, Chunyi Li, Haodong Duan, Zicheng Zhang, and Guangtao Zhai. Affordance benchmark for mllms. arXiv preprint arXiv:2506.00893, 2025. 18
- [135] Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Jiaqi Wang, Haiyang Xu, Ming Yan, Ji Zhang, et al. Amber: An llm-free multi-dimensional benchmark for mllms hallucination evaluation. arXiv preprint arXiv:2311.07397, 2023. 17
- [136] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with mathvision dataset. Advances in Neural Information Processing Systems, 2025. 4, 17
- [137] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 5, 17
- [138] Weiyun Wang, Zhe Chen, Wenhai Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Jinguo Zhu, Xizhou Zhu, Lewei Lu, Yu Qiao, et al. Enhancing the reasoning ability of multimodal large language models via mixed preference optimization. arXiv preprint arXiv:2411.10442, 2024. 17

- [139] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Xixuan Song, Jiazheng Xu, Bin Xu, Juanzi Li, Yuxiao Dong, Ming Ding, and Jie Tang. Cogvlm: Visual expert for pretrained language models. ArXiv, abs/2311.03079, 2023. 17
- [140] Weiyun Wang, Yiming Ren, Haowen Luo, Tiantong Li, Chenxiang Yan, Zhe Chen, Wenhai Wang, Qingyun Li, Lewei Lu, Xizhou Zhu, et al. The all-seeing project v2: Towards general relation comprehension of the open world. arXiv preprint arXiv:2402.19474, 2024. 18
- [141] Weiyun Wang, Min Shi, Qingyun Li, Wenhai Wang, Zhenhang Huang, Linjie Xing, Zhe Chen, Hao Li, Xizhou Zhu, Zhiguo Cao, et al. The all-seeing project: Towards panoptic visual recognition and understanding of the open world. arXiv preprint arXiv:2308.01907, 2023. 18
- [142] Weiyun Wang, Shuibo Zhang, Yiming Ren, Yuchen Duan, Tiantong Li, Shuo Liu, Mengkang Hu, Zhe Chen, Kaipeng Zhang, Lewei Lu, et al. Needle in a multimodal haystack. Advances in Neural Information Processing Systems, 37:20540–20565, 2025. 18
- [143] Wei-Yao Wang, Zhao Wang, Helen Suzuki, and Yoshiyuki Kobayashi. Seeing is understanding: Unlocking causal attention into modality-mutual attention for multimodal llms. arXiv preprint arXiv:2503.02597, 2025. 18
- [144] Xingrui Wang, Wufei Ma, Tiezheng Zhang, Celso M de Melo, Jieneng Chen, and Alan Yuille. Spatial457: A diagnostic benchmark for 6d spatial reasoning of large mutimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24669–24679,

2025. 18

- [145] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 17
- [146] Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569–113697,

2024. 17

- [147] Justus Westerhoff, Erblina Purelku, Jakob Hackstein, Jonas Loos, Leo Pinetzki, and Lorenz Hufe. Scam: A real-world typographic robustness evaluation for multimodal foundation models. arXiv preprint arXiv:2504.04893, 2025. 18
- [148] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, et al. Janus: Decoupling visual encoding for unified multimodal understanding and generation. arXiv preprint arXiv:2410.13848, 2024. 18
- [149] Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Chunyi Li, Wenxiu Sun, Qiong Yan, Guangtao Zhai, et al. Q-bench: A benchmark for general-purpose foundation models on low-level vision. arXiv preprint arXiv:2309.14181, 2023. 15, 17
- [150] Penghao Wu and Saining Xie. V?: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13084–13094, 2024. 18

- [151] Ziheng Wu, Zhenghao Chen, Ruipu Luo, Can Zhang, Yuan Gao, Zhentao He, Xian Wang, Haoran Lin, and Minghui Qiu. Valley2: Exploring multimodal models with scalable vision-language design. arXiv preprint arXiv:2501.05901,

2025. 18

- [152] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218, 2024. 17
- [153] XAI. Grok-1.5 vision preview. 2024. 16
- [154] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024. 4, 16
- [155] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, et al. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215, 2025. 5
- [156] Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. arXiv:2308.16911, 2023. 1
- [157] Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025. 18
- [158] Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, et al. xgen-mm (blip-3): A family of open large multimodal models. arXiv preprint arXiv:2408.08872, 2024. 16
- [159] Le Xue, Ning Yu, Shu Zhang, Artemis Panagopoulou, Junnan Li, Roberto Mart´ın-Mart´ın, Jiajun Wu, Caiming Xiong, Ran Xu, et al. Ulip-2: Towards scalable multimodal pretraining for 3d understanding. In CVPR, pages 27091– 27101, 2024. 1
- [160] Cheng Yang, Chufan Shi, Yaxin Liu, Bo Shui, Junjie Wang, Mohan Jing, Linran Xu, Xinyu Zhu, Siheng Li, Yuxiang Zhang, et al. Chartmimic: Evaluating lmm’s cross-modal reasoning capability via chart-to-code generation. arXiv preprint arXiv:2406.09961, 2024. 17
- [161] Jie Yang, Feipeng Ma, Zitian Wang, Dacheng Yin, Kang Rong, Fengyun Rao, and Ruimao Zhang. Wethink: Toward general-purpose vision-language reasoning via reinforcement learning. arXiv preprint arXiv:2506.07905, 2025. 17
- [162] Zhibo Yang, Jun Tang, Zhaohai Li, Pengfei Wang, Jianqiang Wan, Humen Zhong, Xuejing Liu, Mingkun Yang, Peng Wang, Yuliang Liu, et al. Cc-ocr: A comprehensive and challenging ocr benchmark for evaluating large multimodal models in literacy. arXiv preprint arXiv:2412.02210,

2024. 18

- [163] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. 17

- [164] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models. In The Thirteenth International Conference on Learning Representations, 2024. 17
- [165] Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, et al. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration, 2023. 1
- [166] Zhoutong Ye, Mingze Sun, Huan-ang Gao, Chun Yu, and Yuanchun Shi. Moat: Evaluating lmms for capability integration and instruction grounding. arXiv preprint arXiv:2503.09348, 2025. 18
- [167] Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, et al. Mmt-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask agi. arXiv preprint arXiv:2404.16006,

2024. 2, 14, 17

- [168] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv:2308.02490, 2023. 2, 3, 14
- [169] Jiakang Yuan, Tianshuo Peng, Yilei Jiang, Yiting Lu, Renrui Zhang, Kaituo Feng, Chaoyou Fu, Tao Chen, Lei Bai, Bo Zhang, et al. Mme-reasoning: A comprehensive benchmark for logical reasoning in mllms. arXiv preprint arXiv:2505.21327, 2025. 18
- [170] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multidiscipline multimodal understanding and reasoning benchmark for expert agi. arXiv:2311.16502, 2023. 2, 3, 15
- [171] Bo Zhang, Shuo Li, Runhe Tian, Yang Yang, Jixin Tang, Jinhao Zhou, and Lin Ma. Flash-vl 2b: Optimizing visionlanguage model performance for ultra-low latency and high throughput. arXiv preprint arXiv:2505.09498, 2025. 18
- [172] Ge Zhang, Xinrun Du, Bei Chen, Yiming Liang, Tongxu Luo, Tianyu Zheng, Kang Zhu, Yuyang Cheng, Chunpu Xu, Shuyue Guo, et al. Cmmmu: A chinese massive multidiscipline multimodal understanding benchmark. arXiv preprint arXiv:2401.11944, 2024. 17
- [173] He Zhang, Shenghao Ren, Haolei Yuan, Jianhui Zhao, Fan Li, Shuangpeng Sun, Zhenghao Liang, Tao Yu, Qiu Shen, and Xun Cao. Mmvp: A multimodal mocap dataset with vision and pressure sensors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21842–21852, 2024. 17
- [174] Jieyu Zhang, Weikai Huang, Zixian Ma, Oscar Michel, Dong He, Tanmay Gupta, Wei-Chiu Ma, Ali Farhadi, Aniruddha Kembhavi, and Ranjay Krishna. Task me anything. arXiv preprint arXiv:2406.11775, 2024. 16, 17
- [175] Jianshu Zhang, Dongyu Yao, Renjie Pi, Paul Pu Liang, and Yi R Fung. Vlm2-bench: A closer look at how well vlms implicitly link explicit matching visual cues. arXiv preprint arXiv:2502.12084, 2025. 18

- [176] Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Yu Qiao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? In European Conference on Computer Vision. Springer. 4, 18
- [177] Tianyu Zhang, Suyuchen Wang, Lu Li, Ge Zhang, Perouz Taslakian, Sai Rajeswar, Jie Fu, Bang Liu, and Yoshua Bengio. Vcr: Visual caption restoration. arXiv preprint arXiv:2406.06462, 2024. 16, 17
- [178] Yuhui Zhang, Yuchang Su, Yiming Liu, Xiaohan Wang, James Burgess, Elaine Sui, Chenyu Wang, Josiah Aklilu, Alejandro Lozano, Anjiang Wei, et al. Automated generation of challenging multiple-choice questions for vision language model evaluation. arXiv preprint arXiv:2501.03225,

2025. 18

- [179] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 5
- [180] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 18
- [181] Yi-Fan Zhang, Qingsong Wen, Chaoyou Fu, Xue Wang, Zhang Zhang, Liang Wang, and Rong Jin. Beyond llavahd: Diving into high-resolution large multimodal models. arXiv preprint arXiv:2406.08487, 2024. 17
- [182] Yi-Fan Zhang, Huanyu Zhang, Haochen Tian, Chaoyou Fu, Shuangqing Zhang, Junfei Wu, Feng Li, Kun Wang, Qingsong Wen, Zhang Zhang, et al. Mme-realworld: Could your multimodal llm challenge high-resolution real-world scenarios that are difficult for humans? arXiv preprint arXiv:2408.13257, 2024. 14, 17
- [183] Zicheng Zhang, Ziheng Jia, Haoning Wu, Chunyi Li, Zijian Chen, Yingjie Zhou, Wei Sun, Xiaohong Liu, Xiongkuo Min, Weisi Lin, et al. Q-bench-video: Benchmark the video quality understanding of lmms. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3229–3239, 2025. 17
- [184] Zicheng Zhang, Haoning Wu, Chunyi Li, Yingjie Zhou, Wei Sun, Xiongkuo Min, Zijian Chen, Xiaohong Liu, Weisi Lin, and Guangtao Zhai. A-bench: Are lmms masters at evaluating ai-generated images? arXiv preprint arXiv:2406.03070, 2024. 15, 17
- [185] Tiancheng Zhao, Qianqian Zhang, Kyusong Lee, Peng Liu, Lu Zhang, Chunxin Fang, Jiajia Liao, Kelei Jiang, Yibo Ma, and Ruochen Xu. Omchat: A recipe to train multimodal language models with strong long context and video understanding. arXiv preprint arXiv:2407.04923, 2024. 17
- [186] Xiangyu Zhao, Shengyuan Ding, Zicheng Zhang, Haian Huang, Maosong Cao, Weiyun Wang, Jiaqi Wang, Xinyu Fang, Wenhai Wang, Guangtao Zhai, et al. Omnialign-v: Towards enhanced alignment of mllms with human preference. arXiv preprint arXiv:2502.18411, 2025. 16
- [187] Xiangyu Zhao, Wanghan Xu, Bo Liu, Yuhao Zhou, Fenghua Ling, Ben Fei, Xiaoyu Yue, Lei Bai, Wenlong Zhang, and Xiao-Ming Wu. Msearth: A benchmark for multimodal scientific comprehension of earth science. arXiv preprint arXiv:2505.20740, 2025. 18

- [188] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: A comprehensive benchmark for multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024. 5, 16
- [189] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 4, 5
- [190] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. arXiv preprint arXiv:2411.00836,

2024. 4, 5

- [191] Yuxin Zuo, Shang Qu, Yifei Li, Zhangren Chen, Xuekai Zhu, Ermo Hua, Kaiyan Zhang, Ning Ding, and Bowen Zhou. Medxpertqa: Benchmarking expert-level medical reasoning and understanding. arXiv preprint arXiv:2501.18362, 2025. 18

### A. Benchmarks

In this section, we list all supported benchmarks with a brief introduction and categorize them into multiple categories.

#### A.1. General Capability

- • MMBench Series [80]: The MMBench Series, which includes MMBench and its Chinese version MMBench-CN, serves as a pioneering benchmark for assessing Large Vision-Language Models (LVLMs) across 20 distinct capabilities. MMBench-CN adapts the questions and choices of MMBench into Chinese based on GPT-4. These benchmarks are designed to promote a more precise and comprehensive evaluation of VLMs.
- • SEEDBench & SEEDBench2 [64]: SEEDBench comprises 19,000 multiple-choice questions that span 12 dimensions, while SEEDBench2 expands this dataset to 24,000 questions across 27 dimensions. Both benchmarks provide a robust framework for evaluating the multifaceted capabilities of AI models.
- • MME [33]: a holistic benchmark aimed at evaluating MLLMs. It assesses perception and cognition across 14 subtasks, featuring manually curated instructionanswer pairs to prevent data leakage and ensure a fair comparison without the need for prompt engineering.
- • MMT-Bench [167]: a comprehensive benchmark designed to assess LVLMs across massive multimodal tasks requiring expert knowledge and deliberate visual recognition, localization, reasoning, and planning.
- • MME-RealWorld [182]: a large-scale benchmark for evaluating MLLMs in real-world scenarios. It comprises 29,429 question-answer pairs and encompasses 43 subtasks, providing a rich dataset for testing the practical applications of MLLMs.

#### A.2. Text Recognition & Understanding

- • TextVQA [112]: a dataset for assessing visual reasoning based on textual content within images. TextVQA challenges models to read and reason about the text present in images to answer corresponding questions. To accomplish this, models must integrate the textual information found in the images and utilize it to formulate responses to TextVQA’s queries.
- • OCRVQA [98]: a benchmark for visual question answering by reading text in images. It introduces the OCR-VQA-200K dataset, which consists of over 200,000 book cover images. This study integrates techniques from Optical Character Recognition (OCR) and Visual Question Answering (VQA) to tackle the novel challenges associated with this dataset.
- • OCRBench [82]: a comprehensive evaluation bench-

mark designed to assess the Optical Character Recognition (OCR) capabilities of LMMs across various text-related visual tasks, including text recognition, scene text-centric VQA, document-oriented VQA, key information extraction, and handwritten mathematical expression recognition.

• DocVQA [97]: designed to encourage a “purposedriven” point of view in Document Analysis and Recognition research, where the document content is extracted and used to respond to high-level tasks defined by the human consumers of this information.

#### A.3. Structuralized Content Understanding

- • ChartQA [95]: a benchmark designed to assess question answering skills related to charts, with an emphasis on complex visual and logical reasoning. The dataset comprises 9,600 human-written questions and 23,100 machine-generated questions based on chart summaries. It challenges models to perform intricate reasoning that includes logical and arithmetic operations as well as to analyze visual chart features.
- • InfoVQA [96]: a dataset aimed at enhancing the automatic understanding of infographic images through VQA. It includes a diverse collection of infographics paired with natural language questions and answers, focusing on the need for models to reason across document layouts, textual content, graphical elements, and data visualizations.
- • TableVQA-Bench [59]: a benchmark created for visual question answering focused on table data. It integrates images and question-answer pairs that were not previously available in existing datasets. The benchmark consists of 1,500 QA pairs generated from textformatted tables using a large language model (LLM).
- • SEEDBench2-Plus [62]: a benchmark that includes 2,300 visual questions covering three categories (Charts, Maps, and Webs) and 63 fine-grained data types found in real-world scenarios. It is designed to evaluate the capabilities of multimodal large language models in understanding text-rich visual content.

#### A.4. Subjective (LLM Judge)

- • MM-Vet [168]: an evaluation benchmark for LMMs designed to assess their performance on complex multimodal tasks. It provides a systematic structure for these tasks and proposes unified evaluation metrics. The benchmark utilizes Large Language Model (LLM)-based evaluators to assess a variety of question and answer types, ensuring a comprehensive analysis of LMM capabilities.
- • LLaVABench [79]: a dataset created to evaluate the capabilities of LMMs in more challenging tasks and their ability to generalize to new domains. It con-

sists of a diverse set of 24 images with 60 questions in total, including indoor and outdoor scenes, memes, paintings, sketches, etc., and each image with a highlydetailed and manually-curated description and a proper selection of questions.

#### A.5. Mathematics & Examination

- • MathVista [87]: a benchmark designed to evaluate the mathematical reasoning capabilities of LLMs and LMMs in visual contexts. It comprises 6,141 examples from 28 existing multimodal datasets and three new datasets (IQTest, FunctionQA, and PaperQA), challenging models with deep visual understanding and compositional reasoning.
- • MathVision [87]: a dataset addressing limitations in existing benchmarks for evaluating mathematical reasoning in visual contexts. It includes 3,040 highquality mathematical problems sourced from actual competitions, spanning 16 disciplines and varying in difficulty across five levels.
- • ScienceQA-IMG [88]: a multimodal benchmark for answering science-related questions. It contains approximately 21,000 questions and annotations, promoting reasoning through the Chain of Thought (CoT) approach.
- • MMMU [170]: a comprehensive benchmark for evaluating multimodal models on tasks that require expertlevel knowledge. It features 11,500 questions across six disciplines and 183 subfields.

#### A.6. Low-level & Aesthetics

- • AesBench [53]: a benchmark for evaluating aesthetic perception capabilities of MLLMs with an expertlabeled database and from four shallow-to-deep perspectives.
- • Q-Bench [149]: a benchmark for evaluating MLLMs in low-level vision tasks, including perception, description, and assessment. Only multi-choice questions are included in our implementation.
- • A-Bench [184]: a benchmark for evaluating LMMs in assessing AI-generated images, including 2864 AIGIs from 16 text-to-images models.

#### A.7. Long-Context, Multi-Image & Multi-Turn

- • DUDE [128]: a comprehensive benchmark aimed at advancing Document AI by addressing limitations in current methods for visually-rich documents. The benchmarks include 6315 visual questions, with an average of 5.7 images per question.
- • MMLongBench-Doc [92]: a benchmark designed to evaluate understanding of long-context documents in LVLMs. It consists of 1,091 expert-annotated questions based on 135 lengthy PDF documents, averaging

- 47.5 pages and 21,214 tokens each. This benchmark uniquely requires that answers be derived from various multimodal sources, including text, images, charts, tables, and layout structures, and 33% of the questions require cross-page evidence.
- • SlideVQA [117]: a multi-image document VQA dataset, consisting of 52k+ slide images from 2.6k+ slide decks and 14.5k questions. It requires complex reasoning, including numerical and multihop reasoning, and provides annotated arithmetic expressions for numerical answers.
- • BLINK [38]: a benchmark with 14 challenging visual perception tasks for multimodal LLMs, with around 60% of visual questions consisting of multiple images as inputs.
- • MuirBench [131]: a comprehensive benchmark for multimodal LLMs on multi-image understanding with 12 tasks, 10 relation categories, 11264 images and 2600 questions.
- • MMDU [83]: a benchmark designed to evaluate VLMs capabilities in multi-turn, multi-image dialogues, addressing their limitations in handling realworld, complex conversational scenarios. MMDU features longer dialogues with up to 27 turns and 20 images, challenging current LVLMs.

#### A.8. Hallucination

- • HallusionBench [77]: a diagnostic benchmark for evaluating large visual-language models’ reasoning about image-context relationships, focusing on visual illusions and language hallucinations. Consists of 346 images and 1,129 questions to assess response consistency, failure modes, and logical reasoning.
- • POPE [70]: a benchmark for evaluating object hallucination with three tracks (random, popular, and adversarial) and a total of ∼ 9k cases.

#### A.9. Multilingual

- • MMMB and Multilingual MMBench [116]: MMMB is a multilingual multimodal benchmark with six languages, 15 categories, and 12k questions. Multilingual MMBench extends MMBench-DEV to six languages.
- • MT-VQA [118]: a benchmark designed for multilingual Text-Centric Visual Question Answering, addressing limitations of previous datasets that focus on high-resource languages and rely on translation-based approaches. The work introduces high-quality human annotations across nine diverse languages, avoiding ”visual-textual misalignment” and improving multilingual scene understanding.

#### A.10. Miscellaneous

- • MLLMGuard (Safety) [41]: a multi-dimensional safety evaluation suite for MLLMs, including a bilingual image-text evaluation dataset, inference utilities, and a set of lightweight evaluators.
- • GMAI-MMBench (Medical) [15]: a comprehensive multimodal evaluation benchmark for VLMs in the medical field across multiple datasets, image modalities, and clinical tasks.
- • RealWorldQA (Autonomous Driving) [153]: a benchmark for evaluating real-world spatial understanding of multimodal AI models with over 700 images from various scenarios.
- • COCO Caption (Captioning) [76]: Contains 5,000 samples from the COCO Caption Validation set and prompts VLMs to describe the images.
- • MMStar (Data Contamination) [13]: a visionindispensable multi-modal benchmark addressing issues of unnecessary visual content and data leakage in evaluating LVLMs. Comprises 1,500 curated samples and proposes new metrics for data leakage.
- • TaskMeAnything ImageQA [174]: a benchmark generation engine for creating tailored benchmarks with an extendable taxonomy of visual assets.
- • A-OKVQA [107]: a benchmark for assessing VQA using world knowledge and commonsense reasoning with around 25k crowdsourced questions.
- • VCR-wiki [177]: Visual Caption Restoration (VCR) is a new vision-language task that requires models to restore partially obscured texts in images using pixellevel hints. Unlike traditional tasks that often rely on optical character recognition or masked language modeling, VCR emphasizes the integration of visual, textual, and contextual cues to achieve accurate text restoration.

#### A.11. Video Understanding

- • MMBench-Video [31]: a long-form, multi-shot benchmark with about 600 YouTube videos (30 s–6 min) drawn from 16 everyday domains. It offers roughly 2000 volunteer-written Q&A pairs that probe 26 fine-grained skills and uses a GPT-4 adjudication pipeline for reliable scoring, making it a concise testbed for holistic video understanding.
- • Video-MME [34]: the first “full-spectrum” evaluation for video LLMs, containing 900 clips plus 2700 human-annotated Q&A pairs. Videos span short to hour-long segments across six visual domains, and the benchmark explicitly mixes frames, audio, and subtitles to examine cross-modal reasoning over varied temporal scales.
- • MVBench [68]: targets temporal reasoning by con-

verting 20 classic vision tasks into dynamic versions that cannot be solved with a single frame. Automatic conversion of public annotations into multiple-choice QAs ensures fairness, and early results show mainstream MLLMs still fall short on many perception-tocognition temporal skills.

- • MLVU [188]: focuses on long-video comprehension, gathering clips from 3 min up to 2 h and organising nine tasks that demand both global narrative understanding and fine-grained detail tracking. Initial evaluations (20 models, incl. GPT-4o) reveal that even the best system scores only 64.6%, underscoring the difficulty of sustained reasoning over extended contexts.
- • TempCompass [81]: an benchmark designed to gauge temporal perception. It covers diverse temporal aspects (action granularity, motion speed, event order, attribute change, etc.) and multiple task formats (multichoice, yes/no, caption matching/generation). Carefully crafted “conflicting videos” reduce single-frame shortcuts, offering a stricter measure of whether models truly exploit temporal cues.

### B. Acknowledgements

We make our best effort to list all known contributions to the repository. If some of your contributions are missing, please contact opencompass@pjlab.org.cn.

#### B.1. Contributors with 3+ Major Contributions

We list contributors who have made 3+ significant contributions to the development of VLMEvalKit.

##### Qualified Contributors (2025.02):

- • PhoenixZ810 (Xiangyu Zhao): The contributor helped support WeMath [105], LogicVista [154], MM-AlignBench [186], Video-ChatGPT [93], ChatUniVI [57], and Llama-VID [71].
- • amitbcp (Amit Agarwal): The contributor helped support MUIRBench [131], Phi-3.5 [2], Idefics3 [60], VILA [75], xGen-MM [158], and MVTamperBench [4].
- • czczup (Zhe Chen): The contributor helped support the InternVL Series [19,20,39] (V1.5, Mini-InternVL, V2, etc.).
- • Mor-Li (Mo Li): The contributor helped support LLaVA-OneVision [71], GQA [54], and developed the readthedocs site for VLMEvalKit.
- • mayubo2333 (Yubo Ma): The contributor helped support MMLongBench [92], SlideVQA [117], and DUDE [128].
- • sun-hailong (Hailong Sun): The contributor helped support A-OKVQA [107], Parrot, MMMB, and MTLMMBench [116].
- • Cuiunbo (Junbo Cui): The contributor helped support

OmniLMM-12B, MiniCPM-V Series [163] (V1, V2, V2.5).

- • yfzhang114 (yifan zhang) The contributor helped support Slime [181], MME-RealWorld [182], and Amber [135].
- • runninglsy (Shiyin Lu) The contributor helped support Ovis Series [89].
- • tackhwa (Tack Hwa Wong) The contributor helped support Eagle X [111], Moondream, Kosmos2 [101].
- • Weiyun1025 (Weiyun Wang) The contributor helped support InternVL-COT evaluation, InternVL2-8BMPO, InternVL2.5-MPO [138].
- • Myhs-phz (Peiheng Zhou) The contributor helped support MIA-Bench [104], VizWiz [43], OlympiadBench [46], CMMMU [172].
- • BradyFU (Chaoyou Fu) The contributor helped support Video-MME [35], Vita 1.0 [36], Vita 1.5 [37].
- • OliverLeeXZ (Xiaozhe Li) The contributor helped support Emu3-[Chat/Gen] [145], Moonshot APIs, Grok APIs.

##### Qualified Contributors (2025.06):

- • chenjix (JiXuan Chen): The contributor helped support Humanity’s last exam [102], LLama4 [127], ScreenSpot [22], ScreenSpot-v2 [152] and ScreenSpot-Pro [67].
- • Espere-1119-Song (Enxin Song): The contributor helped support Video-MMLU [114], MovieChat1k [113] and VDC [8].
- • maosong2022 (Song Mao): The contributor helped support MMVP [173], CVBench [126] and CharXiv [146], who also help solved many issues in the community.
- • SYuan03 (Shengyuan Ding): The contributor supports mPLUG-Owl3 [164], MM-IFEval [27], ChartMimic [160] and fixed problems in CreationMMBench [30] evaluation.
- • TianhaoLiang2000 (Tianhao Liang): The contributor supports mega-bench [11], Kimi-VL-A3B [124] and add vllm support for QwenVL/LLama4/InternVL Series Model.
- • zzc-1998 (Zicheng Zhang): The contributor supports Q-Bench [149], A-Bench [184] and Q-Bench Video [183].

- B.2. Full Contributor List Report co-authors are excluded from the below list.

- • echo840: The contributor supports OCRBench [82].
- • tianyu-z, sheryc: The contributor supports VCR [177].
- • TousenKaname: The contributor supports GMAIMMBench [15] and MGM-7B [72].
- • ShuoZhang2003: The contributor supports Monkey [73].
- • PCIResearch: The contributor supports TransCoreM.

- • dylanqyuan: The contributor supports AesBench [53].
- • mary-0830: The contributor supports OmChat [185] and VLM-R1 [109].
- • LZHgrla: The contributor supports LLaVAXTuner [25].
- • bingwork: The contributor supports MMAlaya and MMAlaya2 [85].
- • fitzpchao: The contributor supports ShareCaptioner [12] and CogVLM [139].
- • Isaachhh: The contributor adds custom prompts for Bunny [47].
- • eltociear: The contributor adds the Japanese README.
- • iyuge2: The contributor supports GLM-Vision and updates prompts for CogVLM & GLM4v-9B.
- • BrenchCC: The contributor supports Mantis [55].
- • Ezra-Yu: The contributor fixes an error in the acc calculation.
- • StarCycle: The contributor supports local LLMs as the judge.
- • shenyunhang: The contributor fixes a video inference bug.
- • HugoLaurencon: The contributor fixes Idefics2 MathVista prompt.
- • lerogo: The contributor converts base64 images to memory.
- • lihytotoro: The contributor supports MiniCPM-V-2.6 and MiniCPM-o-2.6 [163].
- • hkunzhe: The contributor supports TableVQA [59].
- • azshue: The contributor fixes an XGen-MM problem.
- • anzhao920: The contributor supports RBDash.
- • youngfly11: The contributor supports COREMM [45].
- • YuZhiyin: The contributor supports Claude3-V.
- • VictorSanh: The contributor adds Idefics2 custom prompts.
- • Jize-W: The contributor fixes an XComposer inference error.
- • YJY123: The contributor supports XVERSE-V-13B.
- • naoto0804: The contributor supports Azure OpenAI API.
- • binwang777: The contributor supports 360VL-70B.
- • KainingYing: The contributor supports MMTBench [167].
- • FeipengMa6: The contributor supports WeMM [26] and WeThink-Qwen2.5VL-7B [161].
- • scikkk: The contributor supports Math-Vision [136].
- • zzc-1998: The contributor supports Q-Bench [149] and A-Bench [184].
- • weikaih04: The contributor supports TaskMeAnything-V1-ImageQA-Random [174].
- • kq-chen: The contributor supports Qwen2-VL [137].
- • zeyofu: The contributor supports BLINK [38].

- • jinyu121: The contributor adds the TextMCQ dataset class.
- • Quakumei: The contributor fixes a README typo.
- • max-yue: The contributor fixes a README typo.
- • dongyh20: The contributor of Ola [84].
- • lrlbbzl: The contributor of URSA-8B and URSA-8BPS-GRPO [90].
- • lyccnb: The contributors of CG-Bench [9].
- • jscslld: The contributors of CG-Bench [9] and CG-AVCounting [86].
- • andrewliao11: The contributor of Q-Spatial Bench [74].
- • wulipc: The contributor of CC-OCR [162].
- • mfarre: The contributor of SmolVLM and SmolVLM2.
- • ttguoguo3: The contributor of CRPE [140,141], MMNIAHBench [142].
- • teowu, Coobiw: The contributors of Aria [66].
- • white2018: The contributor of MiniMonkey [51].
- • LingyiHongfd: The contributor of WorldSense [48].
- • Khang-9966, huynhbaobk: The contributors of VIntern [28].
- • ZhangYuanhan-AI: The contributor of LLaVAVideo [180].
- • CMeteor: The contributor of TeleMM.
- • CaraJ7: The contributor of MathVerse [176] and MME-COT [56].
- • jiutiancv: The contributor of JT-VL-Chat.
- • Baiqi-Li: The contributor of NaturalBench [63].
- • DataWizardLiu: The contributor of DoubaoVL.
- • TobiasLee: The contributor of VLRewardBench [69].
- • ZhiminYao1: The contributor of Taiyi.
- • Haochen-Wang409: The contributor of Ross [133].
- • hills-code: The contributor of Janus-1.3B [148].
- • nbl97: The contributor of HunYuan API.
- • ChuanyangZheng: The contributor of BailingMM API.
- • lcysyzxdxc: The contributor of R-Bench [65].
- • lerogolerogo: The contributor of MMGenBench [50].
- • thomas-yanxin: The contributor of XinYuan-VL-2B.
- • andimarafioti: The contributor of SmolVLM256B/500B.
- • wufeim: The contributor of 3DSRBench [91].
- • yangyue5114: The contributor of MMVet-Hard.
- • gushu333: The contributor of Aquila-VL-2B [40].
- • bobo0810: The contributor of Phi-4 [1].
- • cmatachuan: The contributor of SAIL-VL-1.5 and SAIL-VL-1.6 [29].
- • dellixx: The contributor of Janus-Pro-1B [17].
- • g-h-chen: The contributor of UCSC-VLAAThinker [10].
- • maojialiang: The contributor of Ristretto.
- • RainJamesY: The contributor of VLM2Bench [175].

- • ryf1123: The contributor of VGRP-Bench [106].
- • suencgo: The contributor of Physics [32].
- • suyccc: The contributor of VMCBench [178].
- • sync-yxh: The contributor of Taichu-VLR.
- • tangkexian: The contributor of LEGO-Puzzle [119].
- • tianbinli: The contributor of MedXpertQA [191].
- • waltsun: The contributor of MOAT [166].
- • weiyao-wang: The contributor of AKI [143].
- • xingruiwang: The contributor of spatial457 [144].
- • xwy-bit: The contributor of VisuLogic [157].
- • yangtian6781: The contributor of MMCR [125].
- • zhaomh1998: The contributor of TDBench [49].
- • Hyggge: The contributor of valley2 [151].
- • xjtupanda: The contributor of V ∗ Benchmark [150].
- • wutaiqiang: The contributor of PhyX [110].
- • mxin262: The contributor of OCR-Reasoning [52].
- • Zhouzone: The contributor of MSEarth [187] and OmniEarth [130].
- • An-LanWang: The contributor of WildDoc [129].
- • JunyingWang959: The contributor of A4Bench [134].
- • JiakangYuan: The contributor of MMEReasoning [169].
- • donahowe: The contributor of Video-Holmes [21].
- • rohunagrawal: The contributor of Omni3DBench [94].
- • snowclipsed: The contributor of TallyQA Dataset [3].
- • JonasLoos: The contributor of SCAM Dataset [147].
- • zjh-tsinghua: The contributor of FlashVL [171].
- • nanocm: The contributor of OmniEarth [130] and XLRS-Bench-Lite [132].
- • zhouyiks: The contributor of MMVMBench
- • aarbelle: The contributor of GraniteVision [122].
- • yuzeng0-0: The contributor of VCR Bench [103].
- • Amber0614: The contributor of GOBench.

