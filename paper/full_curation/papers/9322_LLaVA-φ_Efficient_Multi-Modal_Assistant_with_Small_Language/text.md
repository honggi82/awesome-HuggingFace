[Figure 1]

## LLaVA-Phi: Efficient Multi-Modal Assistant with Small Language Model

Yichen Zhu1, Minjie Zhu1,2, Ning Liu1, Zhicai Ou1, Xiaofeng Mou1, Jian Tang1 1Midea Group, 2East China Normal University

# arXiv:2401.02330v4[cs.CV]22Feb2024

### Abstract

In this paper, we introduce LLaVA-ϕ (LLaVA-Phi), an efficient multi-modal assistant that harnesses the power of the recently advanced small language model, Phi-2, to facilitate multi-modal dialogues. LLaVA-Phi marks a notable advancement in the realm of compact multi-modal models. It demonstrates that even smaller language models, with as few as 2.7B parameters, can effectively engage in intricate dialogues that integrate both textual and visual elements, provided they are trained with high-quality corpora. Our model delivers commendable performance on publicly available benchmarks that encompass visual comprehension, reasoning, and knowledge-based perception. Beyond its remarkable performance in multi-modal dialogue tasks, our model opens new avenues for applications in timesensitive environments and systems that require real-time interaction, such as embodied agents. It highlights the potential of smaller language models to achieve sophisticated levels of understanding and interaction, while maintaining greater resource efficiency. The project is available at https://github.com/zhuyiche/llava-phi.

### 1. Introduction

Large vision language models, including Flamingo [1], GPT-4V [30], and Gemini [33], have exhibited remarkable proficiency in executing instructions, engaging in multi-turn dialogues, and handling image-based question-answering tasks. The progression of open-source vision language models has been significantly propelled by the rapid advancement of open-source Large Language Models like LLaMA [34] and Vicuna [5]. These developments primarily focus on leveraging language models with a minimum of 7B parameters, integrated with a vision encoder to enhance visual comprehension. However, this approach often results in increased test time and reduced inference speed, which are less than ideal for time-sensitive or real-time interactive applications, such as autonomous driving and robotics. This leads to an important inquiry: How effectively can small vision-language assistants perform in comparison?

Gemini [33] has blazed a trail for multi-modal models in

mobile technology. Its streamlined variant, Gemini-Nano, boasts 1.8/3.25 billion parameters and is deployable on mobile devices. However, details like the model architecture, training data, and training methodologies remain proprietary and inaccessible to the public. In the realm of small language models, there have been notable advancements: TinyGSM [23], with 2.6 billion parameters, achieves over 80% accuracy on the GSM8k [7] benchmark. Additionally, models such as Phi [13] have demonstrated capabilities in language understanding, commonsense reasoning, and code generation, rivaling larger language models like LLaMA2-7B. This progress underscores the significant strides being made in the efficiency and effectiveness of smaller-scale language models.

In this paper, we introduce LLaVA-Phi, a compact vision-language assistant powered by a small language model. Our work combines the powerful opensourced multi-modal model, LLaVA-1.5 [24], with the best-performing open-sourced small language models, Phi2 [21]. We follow a two-stage training pipeline and leverage high-quality visual instruction tuning data from LLaVA. LLaVA-Phi was evaluated across eight diverse benchmarks. Despite possessing only 3 billion parameters, it achieves performance comparable to, or even surpassing, some larger multi-modal models that are three times larger. Notably, LLaVA-Phi-3B demonstrates exceptional proficiency in ScienceQA [28], outperforming existing large multimodal models. Additionally, we qualitatively demonstrate LLaVA-Phi’s strong generalization ability in handling challenging questions, generating code based on instructions, and solving mathematical problems.

### 2. Related Work

The rapid advancements in Large Language Models (LLMs) have significantly propelled the development of vision-language models based on LLMs. These models, representing a departure from the capabilities of the preLLM era, are equipped with advanced question-answering and visual comprehension skills. This progress is enabled by using LLMs as language encoding modules. Notable research in this domain includes the LLaVA-family [24, 25, 26, 32], the BLIP-family [8, 20], MiniGPT-4 [37], and oth-

###### Tricky Question

[Figure 2]

User Explain why this meme is funny.

LLaVA-1.5

The meme is funny because it features a dog lying on a wooden floor with a caption that reads "Monday. Just... Monday." The dog's expression and the caption create a humorous and relatable situation for many people who often feel tired or unenthusiastic about starting a new week. The dog's presence in the meme adds a lighthearted and amusing touch to the otherwise mundane topic of Monday morning.

-13B

LLaVA-Phi The meme is funny because it combines the image of a dog lying on the floor with the phrase "Just Monday." The juxtaposition of the dog's relaxed and sleepy state with the phrase "Just Monday" creates a humorous contrast. The dog's laid-back demeanor is often associated with the feeling of Monday morning, which is often considered the start of a new workweek and can be associated with stress or fatigue. The image of the dog lying on the floor adds a lighthearted and unexpected element to the meme, making it amusing and entertaining.

Figure 1. LLaVA-Phi is adept at identifying and responding to complex questions with empathetic reasoning.

ers. Each has demonstrated significant advancements in managing visual-centric dialogues. However, a common limitation of these open-sourced Vision-Language Models (VLMs) is their substantial computational demands, typically ranging from 7B to 65B parameters. This requirement poses challenges for deployment on edge or mobile devices, especially in real-time applications. Gemini [33], a leader in this field, has released three versions of visionlanguage models, including the compact Gemini-Nano with

- 1.8B/3.25B parameters, tailored for smartphones. However, their models and data are not open-sourced. Another initiative, MobileVLM [6], has developed mobileLLaMA with
- 2.7B parameters to facilitate smaller vision-language models. Our paper explores and demonstrates the effectiveness of integrating vision-language models with open-sourced, smaller language models, assessing their potential and efficiency in a variety of applications.

###### Code Generation

[Figure 3]

User

Convert the table in the image to a bar chart using python

import pandas as pd # Read the table data df = pd.read_table(image_path, sep=‘\t’) # Print the DataFrame print(df) ```python

LLaVA-1.5

-13B

LLaVA-Phi

import pandas as pd import matplotlib.pyplot as plt

# Read the table data from the image table_data = pd.read_table(image=True, file_name='table1.png’)

et the title and axis labels

# Create a bar chart plt.bar(table_data['Letter'], table_data['Score’])

# Set the title and axis labels # Add labels and title plt.title('Table 1: Capture for Table with Student Scores’) plt.xlabel('Letter’) plt.ylabel('Score’)

# Display the chart plt.show()

Figure 2. LLaVA-Phi can generate useful codes based on visual input and commands.

### 3. LLaVA-Phi

Our overall network architecture is similar to LLaVA-1.5. We use the pre-trained CLIP ViT-L/14 with a resolution of 336x336 as the visual encoder. A two-layer MLP is adopted to improve the connection of the visual encoder and LLM.

#### 3.1. Training

Supervised fine-tuning on Phi-2. The publicly released Phi-2 model has not undergone fine-tuning. Previous research indicates that even a small amount of high-quality data can significantly enhance performance in areas such as mathematics, language reasoning, and coding tasks. In light of this, we employed supervised fine-tuning to further train Phi-2 using a select set of premium data. This data was organized in the Vicuna format. For our Supervised

Table 1. Multi-modal evaluation on multi-modal benchmarks. Benchmark names are abbreviated due to space limits. VQAv2 [12]; GQA [16]; VizWiz [14]; SQAI: ScienceQA-IMG [28]; VQAT: TextVQA [31]; POPE [22]; MME [10]; MMB: MMBench [27]; SEED: SEED-Bench [18]; MM-Vet [36].

Method LLM VQAv2 VizWiz SQAI VQAT POPE MME MMB MMVet Gemini-Nano2 [33] N/A (3.25B) 67.5 - - 65.9 - - - OpenFlamingo [2] MBT (7B) - - - 33.6 - - 4.6 BLIP-2 [20] Vicuna (13B) 41.0 19.6 61.0 42.5 85.3 1293.8 - 22.4 InstructBLIP [8] Vicuna (7B) - 34.5 60.5 50.1 - - 36.0 26.2 InstructBLIP [8] Vicuna (13B) - 33.4 63.1 50.7 78.9 1212.8 - 25.6 MiniGPT-4 [37] Vicuna (7B) - - - - - 581.7 23.0 22.1 Shikra [4] Vicuna (13B) 77.4 - - - - - 58.8 Otter [19] LLaMA (7B) - - - - - 1292.3 48.3 24.6 Qwen-VL [3] Qwen (7B) 78.8 35.2 67.1 63.8 - - 38.2 Qwen-VL-Chat [3] Qwen (7B) 78.2 38.9 68.2 61.5 - 1487.5 60.6 IDEFICS-9B [17] LLaMA (7B) 50.9 35.5 - 25.9 - - 48.2 IDEFICS-80B [17] LLaMA (65B) 60.0 36.0 - 30.9 - - 54.5 LLaMA-Adapter-v2 [11] LLaMA (7B) - - - - - 1328.4 39.5 31.4 LLaVA [25] Vicuna (7B) - - - - - 502.8 36.2 28.1 LLaVA-1.5 [24] Vicuna (7B) 78.5 50.0 66.8 58.2 85.9 1510.7 64.3 30.5 MobileVLM [6] M-LLaMA (2.7B) - - 61.0 47.5 84.9 1288.9 59.6 LLaVA-Phi Phi-2 (2.7B) 71.4 35.9 68.4 48.6 85.0 1335.1 59.8 28.9

##### Math

[Figure 4]

User Solve the math equation in the image.

The math equation in the image involves adding two numbers, 6 and 22, and then multiplying the result by 2. To solve this problem, we first add 6 and 22, which gives us 28. Then, we multiply 28 by 2, which results in 56. Therefore, the answer to the math problem is 56.

LLaVA-1.5

-13B

LLaVA-Phi The answer to the math equation in the image is 1.

Figure 3. LLaVA-Phi is capable of performing accurate OCR on mathematical equations and solving them correspondingly..

Fine-Tuning (SFT) data, we utilized ShareGPT from an open-source platform. The training was conducted over two epochs, beginning with an initial learning rate of 3e-5, which was linearly decreased over time. Our findings suggest that while this step might be optional, applying SFT to Phi-2 does result in modest improvements across most benchmarks.

Training LLaVA-Phi. Our training approach follows the pipeline used for LLaVA1.5, consisting of a pretraining stage and a subsequent instruction tuning phase.

Initially, we kept the vision encoder and Phi-2 static, focusing exclusively on training the efficient projector. This step is followed by a comprehensive fine-tuning of both the projector and the language model (LLM), aiming to enhance their capabilities in visual comprehension and language processing.

For pre-training, we utilize a filtered subset of the CC-595K dataset [24] over one epoch, applying an initial learning rate of 1e-3 and a batch size of 256. Then, we finetune the model on LLaVA-Instruct-150K dataset for 1 epoch at a learning rate of 2e-5 and a batch size of 256. We implement a weight decay of 0.1 and utilize the Adam optimizer, characterized by momentum parameters of 0.9 and 0.98, and an epsilon value of 1e-7. We fine-tune all parameters in LLM instead of using LoRA.

Computational Cost. Similar to LLaVA1.5, our training process is structured in two stages. For LLaVA-Phi, the pretraining phase takes 1.5 hours, followed by 8 hours dedicated to visual instruction tuning, utilizing 8 A100 GPUs. The integration of techniques such as LoRA [15] and QLoRA [9] has the potential to significantly reduce training time, a possibility we plan to explore in future work.

#### 3.2. Qualitative Results

We present several examples that demonstrate the remarkable generalization capabilities of LLaVA-Phi, comparing its outputs with those of the LLaVA-1.5-13B models. In

Figure 1, a meme is displayed, and we ask the visionlanguage assistant to explain why this meme is considered humorous. While LLaVA-1.5-13B provides a reasonable interpretation based on the image, LLaVA-Phi’s response is more empathetic, highlighting the humor by associating the dog’s ’laid-back demeanor’ with the ’stress or fatigue’ typically associated with a ’new workweek’.

In the second example, we instructed the model to generate Python code for converting an Excel table into a bar chart, as illustrated in Figure 2. LLaVA-1.5-13B generated a simplistic code snippet that only reads the table and prints it, diverging from the instructions to create a plot. In contrast, LLaVA-Phi accurately comprehended the task, providing instructions to read the table, add a title and labels, and correctly plot the bar chart using matplotlib. We believe this enhanced code generation capability stems from Phi-2, which was pre-trained on a large corpus of code snippets and is primarily used for code generation.

The third challenge involves solving a simple math problem, requiring the model to accurately recognize text through OCR and then perform the necessary mathematical computations, as shown in Figure 3. LLaVA-1.5-13B, while providing a step-by-step computation based on the image, incorrectly recognized the numbers and mathematical symbols. In contrast, our proposed LLaVA-Phi, without providing a chain-of-thought reasoning, still produces the correct answer. Our quantitative results on ScienceQA further confirm that LLaVA-Phi excels in these types of questionanswering tasks.

### 4. Experiments

We rigorously evaluated LLaVA-Phi using an extensive array of academic benchmarks specifically designed for multi-modal models. These included tests for general question-answering such as VQA-v2 [12], VizWizQA [14], ScienceQA [28], and TextQA [31], as well as more specialized assessments like POPE [22] for evaluating object hallucination, and MME [10], MMBench [27], and MMVet [36] for a comprehensive evaluation of diverse multi-modal abilities, such as visual understanding and visual commonsense reasoning.

These benchmarks are meticulously structured to challenge and scrutinize complex multi-modal tasks. We benchmarked LLaVA-Phi against a variety of state-of-the-art, large vision-language models, as detailed in Table 1. It is important to note that both our method and LLaVA1.5 utilize the same publicly available datasets for pre-training and visual instruction fine-tuning.

Our model demonstrated a capacity for visual-based question-answering, surpassing many existing large multimodal models. Remarkably, LLaVA-Phi outperformed models that use 7B-parameter or larger Large Language Models (LLMs) as their backbone, such as IDEFICS [17]

and InstructBLIP [8]. A particularly notable achievement was our model’s best performance on ScienceQA [28]. We attribute this success to the Phi-2 language model, which has been specifically trained on code generation and mathematical corpora, thereby enhancing our multi-modal model’s prowess in math-based question-answering.

In the comprehensive multi-modal benchmark of MMBench [27], LLaVA-Phi showed significantly superior performance compared to many existing 7B-LLM-based vision-language models. For example, our model outperformed Otter by 11.5% and InstructBLIP by 23.8%. This underscores the effectiveness of LLaVA-Phi in handling complex multi-modal tasks, reinforcing the potential of smaller, more efficient models in the rapidly evolving field of multi-modal models.

We also compared to MobileVLM [6], a concurrent work that builds up an efficient vision-language model. Across all five benchmarks, our LLaVA-Phi consistently outperforms their method. It’s important to note that the margins of lead are modest, with the exception of ScienceQA. We attribute this performance disparity primarily to the differences in the pretraining stages of the language models.

### 5. Conclusion, Limitation, and Future Works

We introduce LLaVA-Phi, a vision language assistant developed using the compact language model Phi-2. Our work demonstrates that such small vision-language models can perform effectively on standard benchmarks when combined with the LLaVA training methodology and a select dataset of high-quality data. The primary goal of our project is to aid the community in creating lightweight, multi-modal models capable of vision-language reasoning, optimized for operation on edge devices. This innovation paves the way for deploying multi-modal assistants in time-sensitive applications, such as robotics [35, 38].

Limitations. Given that Phi-2 utilizes the codegenmono [29] tokenizer and our model has not been specifically fine-tuned for following multilingual instructions, our LLaVA-Phi architecture is unable to process instructions in multiple languages, including Chinese.

Future Works. As language models have become significantly smaller in size compared to traditional vision-language models, they have become more accessible and affordable for the research community to explore fundamental concepts in vision-language integration. In future work, we plan to examine the impact of the size of the visual encoder and refine the training strategies for small language models, including approaches like direct preference optimization and RLHF, among other techniques. These efforts aim to further reduce model size while enhancing performance.

### References

- [1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736,

2022. 1

- [2] Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An opensource framework for training large autoregressive visionlanguage models. arXiv preprint arXiv:2308.01390, 2023. 3
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 3
- [4] Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023. 3
- [5] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. See https://vicuna. lmsys. org (accessed 14 April 2023), 2023. 1
- [6] Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, et al. Mobilevlm: A fast, reproducible and strong vision language assistant for mobile devices. arXiv preprint arXiv:2312.16886, 2023. 2, 3, 4
- [7] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 1
- [8] W Dai, J Li, D Li, AMH Tiong, J Zhao, W Wang, B Li, P Fung, and S Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023. 1, 3, 4
- [9] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314, 2023. 3
- [10] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023. 3, 4
- [11] Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010,

2023. 3

- [12] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating

- the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017. 3, 4
- [13] Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio C´esar Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, et al. Textbooks are all you need. arXiv preprint arXiv:2306.11644, 2023. 1
- [14] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617,

2018. 3, 4

- [15] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3
- [16] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 3
- [17] H Laurenc¸on, L Saulnier, L Tronchon, S Bekman, A Singh, A Lozhkov, T Wang, S Karamcheti, A Rush, and D Kiela. Obelisc: An open web-scale filtered dataset of interleaved image-text documents. arXiv preprint arXiv:2306.16527,

2023. 3, 4

- [18] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 3
- [19] Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726, 2023. 3
- [20] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 1, 3
- [21] Yuanzhi Li, S´ebastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report. arXiv preprint arXiv:2309.05463, 2023. 1
- [22] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023. 3, 4
- [23] Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen, Rachel Ward, and Yi Zhang. Tinygsm: achieving¿ 80% on gsm8k with small language models. arXiv preprint arXiv:2312.09241, 2023. 1
- [24] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023. 1, 3
- [25] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

2023. 1, 3

- [26] Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li, Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu, et al. Llava-plus: Learning to use tools for creating multimodal agents. arXiv preprint arXiv:2311.05437, 2023. 1
- [27] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023. 3, 4
- [28] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521,

2022. 1, 3, 4

- [29] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. arXiv preprint arXiv:2203.13474, 2022. 4
- [30] OpenAI. Gpt-4 technical report. arXiv preprint, 2023. 1
- [31] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 3, 4
- [32] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf. arXiv preprint arXiv:2309.14525, 2023. 1
- [33] Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1, 2, 3
- [34] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1
- [35] Junjie Wen, Yichen Zhu, Minjie Zhu, Jinming Li, Zhiyuan Xu, Zhengping Che, Chaomin Shen, Yaxin Peng, Dong Liu, Feifei Feng, et al. Object-centric instruction augmentation for robotic manipulation. arXiv preprint arXiv:2401.02814,

2024. 4

- [36] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 3, 4
- [37] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models. arXiv preprint, page arXiv:2304.10592, 2023. 1, 3
- [38] Minjie Zhu, Yichen Zhu, Jinming Li, Junjie Wen, Zhiyuan Xu, Zhengping Che, Chaomin Shen, Yaxin Peng, Dong Liu, Feifei Feng, et al. Language-conditioned robotic ma-

nipulation with fast and slow thinking. arXiv preprint arXiv:2401.04181, 2024. 4

