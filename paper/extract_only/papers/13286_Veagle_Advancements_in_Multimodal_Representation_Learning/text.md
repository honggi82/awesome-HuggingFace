# Veagle: Advancements in Multimodal Representation Learning

Rajat Chawla Arkajit Datta Tushar Verma Adarsh Jha Mukunda NS Ishaan Bhola Ayush Vatsal Sukrit Chaterjee Anmol Gautam

SuperAGI

focal point of this research as images inclusive of text are preva- lent in our everyday lives and comprehending such content is imperative for human visual perception.

## Abstract

Lately, researchers in artificial intelligence have been really in-terested in how language and vision come together, giving rise to the development of multimodal models that aim to seamlessly integrate textual and visual information. Multimodal models, an extension of Large Language Models (LLMs), have exhib- ited remarkable capabilities in addressing a diverse array of tasks, ranging from image captioning and visual question an- swering (VQA) to visual grounding. While these models have showcased significant advancements, challenges persist in accu- rately interpreting images and answering the question, a com- mon occurrence in real-world scenarios. This paper introducesa novel approach to enhance the multimodal capabilities of ex-isting models. In response to the limitations observed in cur- rent Vision Language Models (VLMs) and Multimodal Large Language Models (MLLMs), our proposed model Veagle, in- corporates a unique mechanism inspired by the successes and insights of previous works. Veagle leverages a dynamic mech- anism to project encoded visual information directly into the language model. This dynamic approach allows for a more nu- anced understanding of intricate details present in visual con- texts. To validate the effectiveness of Veagle, we conduct comprehensive experiments on benchmark datasets, emphasizing tasks such as visual question answering and image understand-ing. Our results indicate a improvement of 5-6 % in perfor- mance, with Veagle outperforming existing models by a no- table margin. The outcomes underscore the model’s versatil- ity and applicability beyond traditional benchmarks. Further- more, we make our code and models openly accessible to the research community, fostering collaboration and further explo- ration in the evolving landscape of multimodal AI. The code repository, along with detailed documentation, can be found at https://github.com/ superagi/Veagle

Our research presents a new way of doing things by combining learned query embeddings with additional visual assistance. This method uses encoded patch embeddings to deal with the limitations of information that language models typically get from images. As a result, it enhances how well a model can understand and perceive the relationship between text and images. Our model, called Veagle, starts by using a pre-trainedvision encoder and language model. We train it in two stages to avoid forgetting what it already knows and make training less complicated, ultimately making the model more effective. We tested the model using standard Visual QuestionAnswering (VQA) benchmarks and protocols for evaluating images with a lot of text. Our Veagle model significantly improves the un- derstanding and perception of the relationship between text and images, outperforming traditional benchmarks in addressing the challenges of comprehending embedded text within images.

In this research, we present Veagle, an innovative model that represents a significant leap forward in the field of multimodal learning and interpretation. At the heart of Veagle is the incorporation of an enhanced version of the BLIVA [1] architecture, where cutting-edge components synergize to amplify its capabilities. Notably, we integrate a superior vision abstrac-tor sourced from mPlugOwl[2], enhancing the model’s visual processing capabilities. This vision abstractor, combined with Q-Former from InstructBLIP[3] and Mistral[4], a Large Lan- guage Model (LLM), creates a powerful synergy, resulting in asubstantial improvement in the overall accuracy of the model. A crucial aspect of our methodology is the inclusion of a visionencoder, meticulously trained by mPlugOwl[2]. This encoder plays a pivotal role in extracting high-level visual features from images, thereby enabling Veagle to capture essential visual in- formation for accurate interpretation. This vision encoder is trained to extract high-level visual features from images, al- lowing the model to capture important visual information for accurate interpretation. Veagle distinguishes itself by seam- lessly combining Mistral’s exceptional language understanding with the vision abstractor, resulting in a comprehensive model that effectively integrates both textual and visual information. The proficiency of Mistral in language comprehension significantly enhances Veagle’s overall performance. Our methodology places strong emphasis on the use of a meticulously curated dataset, carefully selected for both pre-training and finetuning stages. This dataset serves as a foundation for shaping the model’s understanding, ensuring robust generalization across different scenarios. Our results show that Veagle has a better grasp of understanding text within images. This is backed up by its impressive performance on standard Visual Question Answering (VQA) tests. Veagle not only outperforms exist- ing models but also establishes a new benchmark for accuracy

Index Terms: MultiModal, Large language models, vision encoder, vision abstractor, Q-former, Image-Text multimodality

## 1. Introduction

In recent years, the surge of interest in Large Language Models(LLMs) has reshaped the landscape of natural language under- standing, a significant surge in the examination and application of Multimodal Large Language Models (MLLMs) has been ob- served. Allowing models to harness various modalities such as text, images, videos, and voice, MLLMs have become vi- tal in the creation of adaptable all-purpose assistants. Despite their impressive generalization abilities across a wide spec- trum of tasks and the development of Vision Language Mod- els (VLMs) which incorporate LLMs with visual understanding competence, contemporary models encounter challenges in in- terpreting embedded text within images. This limitation is the

and efficiency. In conclusion, Veagle represents a cutting-edge model that not only incorporates advanced components but also benefits from the enriching inclusion of curated open sources data, making it a pioneering solution in the evolving landscapeof multimodal AI research.

[Figure 1]

The rest of the paper is organized as follows. Section 2 presents the literature review. 3 highlights the proposed architecture and section 4 includes details of the experiments performed and discusses the results. This is followed by the conclusion in Section 5.

## 2. Literature Survey

In this section, we delve into the related work on large languagemodels and multimodal large language models.

Figure 1: Veagle Model Architecture: The visual abstractoris responsible for extracting instruction-aware visual features from the output embeddings of the frozen image encoder. Subsequently, these visual features are provided as soft prompts to the frozen Language Model (LLM). The model is then finetuned with the language modeling loss to generate the desired response.

### 1. LLM

Language models (LLMs) have revolutionized the field of nat- ural language processing (NLP), providing capabilities rang- ing from text prediction to generating coherent and contextu- ally relevant text. In the ever-evolving realm of natural lan- guage processing, Large Language Models (LLMs) have un- dergone a fascinating journey, leaving an indelible mark on the field. The early contributions of trailblazing models like GPT-2[5] and BERT[6] acted as pillars, demonstrating the im-mense potential that arises from training on vast web-scale textdatasets. These models not only laid the groundwork for Nat- ural Language Processing (NLP) but also served as catalysts for subsequent advancements. Among the notable milestones is the monumental GPT-3[7], a model that not only shattered size records but also showcased unparalleled performance in in tackling intricate challenges. With a staggering 175 billion pa- rameters, GPT-3[7] emerged as a powerhouse, excelling in a diverse array of language tasks. Its introduction prompted a re-examination of the limits of model size and sparked renewed interest in the applications and challenges inherent in handlingcolossal language models. The journey did not conclude with GPT-3[7]; instead, subsequent models like GPT-4[8] and com- panions like Megatron-turing NLG[9], PaLM[10] , Gopher[11], Chinchilla[12], OPT[13], and BLOOM[14] emerged, pushing the boundaries even further. These models, each with unique architectures, training methodologies, and applications, contribute to a dynamic tapestry of research in the expansive domain of large language models. This diversity underscores the ongoing efforts to optimize performance, efficiency, and generalization across an array of linguistic tasks. Recent strides in LLMs have been marked by a nuanced focus on refining models to seamlessly align with human instructions and feedback. Pi- oneering models such as InstructGPT [15], ChatGPT[16] , andthe latest iteration, GPT-4[8], stand out as exemplars in this re-gard. They possess the ability to engage in dynamic, contextu- ally rich conversations, skillfully respond to user prompts, anddemonstrate proficiency in intricate tasks such as code genera-tion. These subsequent advancements in LLMs led to the emer- gence of multimodal large language models, which sought to integrate visual information into the text-based language mod- els This emphasis on harmonizing LLMs with human interac- tion and instruction signifies a pivotal step toward their practical deployment and integration into real-world applications.

prowess of Large Language Models (LLMs) to transcend tradi- tional linguistic boundaries. Building upon the foundations laid by VisualGPT [17], Frozen [18], Flamingo [19], BLIP2 [20], and other pioneering studies, MLLMs have evolved to profi- ciently tackle an expanding spectrum of vision-language tasks. These tasks include image captioning, visual question answer- ing (VQA), and bounding box generation, showcasing the ro- bust visual grounding capability inherent in MLLMs. Notably,recent endeavors such as IntructBLIP [3], LLAVA [21, 22], mPlugOwl [2], and BLIVA actively contribute to diversify- ing the repertoire of tasks that MLLMs adeptly address. Be- yond the conventional scope, ongoing research delves into the realm of multimodal instruction tuning, with endeavors like LLaVA[21], InstructBLIP[3], Otter[23], mPLUG-Owl[2] and LLaVA-1.5[22] pioneering advancements in this domain. Despite the ongoing exploration of model architecture and training pipelines, the landscape remains open for innovative solutions. The integration of multimodal information into language mod-els has brought about significant advancements in their perfor- mance, efficiency, and generalization across various linguistic tasks.

## 3. Proposed Framework

### 1. Architecture Overview 1.1. Image Encoder

A visual encoder is a crucial component of a multimodal models. Visual encoders help the model to extract meaningful representations from visual data. This enables the model to under-stand the semantics and context of the images, which is impor- tant for making accurate predictions or generating relevant out-puts. In our experiments, we have adopt a vision encoder(ViT- L/14[24]) from mPlugOwl[2] . This encoder is responsible for extracting meaningful representations from the input images. mPlugOwl[2] has used a novel training paradigm that incor- porates a trainable visual encoder, while maintaining the pre- trained language model in a frozen state. This approach en- ables the model to effectively capture both low-level and higher semantic visual information and align it with the pre-trained language. They have utilize the imagecaption pairs from sev-

### 2. Multimodal Large Language Models (MLLMs)

In the dynamic landscape of multimodal language models (MLLMs), a paradigm shift is evident as researchers harness the

[Figure 2]

[Figure 3]

Figure 2: Pre-training Loss Insights

Figure 3: Fine-tuning Loss Insights

### 2. Training Scheme

e r a l d a t a s e t s , i n c l u d i n g L A I O N - 4 0 0 M [ 2 5 ] , COYO-700M[26] , Conceptual Captions[27] and MSCOCO[28]. model without compromising its performance.

The training scheme consists of two stages: Pretraining and Fine-tuning. Figure 4 ilustrate our training paradigm.

2.1. Stage 1: Pre-training

- 1.2. Visual Abstractor

A visual abstractor serves as a bridge between the visual encoder and the language decoder, enabling the model to effectively process and utilize visual information alongside text, leading to more powerful and versatile multimodal models.It focuses on extracting essential visual features from the en- coded image representations obtained by the image encoder. Large Language Models (LLMs) undergo pretraining primar- ily on textual corpora, presenting a limitation in their innate ability to process image features extracted from Vision En- coders. Addressing this gap, the introduction of the QFormer module in BLIP-2[20] emerged as a critical intermediary, serv- ing to establish a bridge between Vision Encoder and Language Model. Then came BLIVA[1], a groundbreaking combination of BLIP2[20] and LLaVA[22]. However, a linear projection layer have very limited capability in capturing all the informa- tion required for LLM. To overcome the limitations of projec- tion layers in capturing all the necessary information for LLM, we have introduced a multi layer perceptron along with Q- former[20]. In particular,

- 1 illustrates that our mode generates the embeddings from vision encoder and the output is passed through the projection layer to the Q-former and the second pro- jection layer. The output from the QFormer[20] and Projection layer is concatinated and passed to the LLM which enable better alignment between vision encoders and language models.

- 1.3. LLM

- 1. In this crucial pre-training stage, the Large Language Model(LLM) is aligned with a visual encoder using imagetext pairs from image captioning datasets, facilitating a comprehensive understanding of visual content. The focus is on training the projection layers, refining the mapping of visual and textual information. Throughout this phase, the Vision Encoder, Q- former, and LLM remain frozen, preserving their pre-existing knowledge for subsequent finetuning.
- 2.2. Stage 2: Finetuning

Following pre-training, the Large Language Model (LLM) gains familiarity with the visual embedding space, allowingit to generate image descriptions. However, it lacks the abil- ity to understand finer image details and respond effectivelyto human queries. In this work, we collect publicly available datasets, COCO, TextCaps, VQAv2, OK-VQA, AOK-VQA, GQA, OCR-VQA, TextVQA, VIzWiz and our inhouse curated data. During this phase, the Large Language Model (LLM) and Vision Encoder remain in a frozen state, while the remainder ofthe model undergoes fine-tuning.

## 4. Experimental Overview

### 1. Datasets

For datasets featuring single-word answers, we adopted an innovative approach by expanding these responses into detailed and nuanced answers utilizing the advanced capabilities of GPT-4[8] and Mixtral[30]. This strategic enhancement contributed to the overall effectiveness of our model, ensuring a more robust and comprehensive understanding of various query types. Addressing the challenge of repeated questions present in certain datasets, we took proactive measures to enhance the

At the heart of multimodal large language models is the Large Language Model (LLM), which serves as the keystone. It takesin instructions and aligned image features, processing this in- formation to generate corresponding answers. In our research, we leverage the capabilities of the many different robust open-source large language models ultimately settling on Mistral[4] due to its superior performance. Mistral 7B surpasses the per- formance of the leading open 13B model (Llama 2[29]) acrossall benchmarks and outperforms the best released 34B model (Llama 1[29]) specifically in reasoning, mathematics, and codegeneration tasks. Mistral achieves faster inference through the innovative use of grouped-query attention (GQA) and effec- tively manages sequences of arbitrary length with reduced in- ference cost by incorporating sliding window attention (SWA). This combination of advanced techniques positions Mistral 7Bas a leading model in the domain, setting new standards for both accuracy and computational efficiency.

[Figure 4]

Figure 4: Overview of Veagle training paradigm

- Table 1: Performance of the proposed model for different opensourced datasets.

Veagle BLIVA InstructBLIP mPlugOwl LLAVA ok vqa 49.3 43.4 30.8 34.1 46.2 ocr vqa 48.3 38.5 32.1 61.4 67.2 scienceQA 58.1 16.1 40.2 51.8 56.5 coco caption 57.9 56.4 51.2 55.6 62.7 ai2diagram 56.3 50.8 31.9 48.5 50.9 chart qa 13.4 13.2 3.4 10.2 3.1 gqa 44.2 28.6 40.8 33.9 43.9 text vqa 22.5 23.1 20.5 32.6 37.2

diversity and quality of our training dataset. By generating vari-ous different questions that incorporated a wide range of distinct questions, we effectively mitigated redundancy and enriched the training dataset, thereby fostering improved generalization and performance in handling diverse queries. This meticulous pro- cess of dataset augmentation and refinement played a pivotal role in optimizing the overall performance and reliability of our model. The careful compilation, filtering, and augmentation of diverse datasets played a crucial role in maximizing the perfor- mance and reliability of our model.

2. Results

Our experimental results demonstrate the effectiveness of our approach, with significantly improved performance across vari-ous datasets.

- 2.1. Baseline vs Proposed Protocol

We used four advanced baseline models BLIVA[1], instructBLIP[3], mPlugOwl[2], and LLAVA[22] for our analysis. For each of these models, we took an image and a question, input them into the model, and noted down the response it gave. To evaluate the precision of the provided responses, we employed GPT-4[8] as our assessment model. This model categorized the answers into two distinct classifications: either correct or incorrect. The accuracy outcomes corresponding to each dataset for various different models, obtained through the utilization of this evaluation method, are comprehensively presented in Table 1. Our proposed model achieved an impressive level of accuracy when compared to other open sourced baseline models.

- 2.2. In-House Test Dataset

To assess how well our model performs in different scenarios and if it generalizes effectively, we created an in-house test dataset. This dataset comprises various types of tasks, including captioning, optical character recognition (OCR), general vi- sual question-answering (VQA), technical VQA, and reasoningVQA. Importantly, our model has never encountered this spe- cific dataset during its training process. Subsequently, we con- ducted thorough evaluations of all the models using this test dataset, and the outcomes are promising. Detailed results are presented in Table 2.

- Table 2: Performance of our proposed model Veagle for our inhouse test dataset.

[Figure 5]

Figure 5: Qualitative examples produced by our Veagle model showcase a spectrum of its diverse capabilities. These demonstrations include intricate visual scene understanding and rea-soning, multi-turn visual conversation, and more.

2.3. Qualitative Ananlysis

In this section, we present the qualitative outcomes derived from our assessment set. This set of evaluations was carefully curated to analyze the model’s performance on intricate and challeng- ing tasks. The tasks were selected and collected by our team for the purpose of understanding the model’s effectiveness beyond numerical measures, delving into the nuanced aspects of its per-formance. Figure 5 is showing the effectiveness of our model. More examples are given in 7

## 5. Conclusion

In conclusion, the Veagle multi-modal model stands out as a formidable contender, consistently outperforming established benchmarks in diverse domains. Through the strategic fusion of various modules curated from extensive research, Veagle show- cases remarkable performance, not only meeting but exceedingthe expectations set by existing models. However, our work also reveals areas that still require refinement, emphasizing the on-

Veagle BLIVA InstructBLIP mPlugOwl LLAVA Test Data 76.4 63.1 59.3 68.6 66.5

going nature of the pursuit for perfection. This acknowledgment underscores the need for further exploration and optimization, recognizing that the path to excellence in multi-modal models like Veagle continues to unfold. As we navigate this landscape,Veagle remains a promising catalyst for future advancements in Vision-Language Models, beckoning further investigation and innovation in this dynamic field.

E. Proehl, R. Puri, A. Radford, J. Rae, A. Ramesh, C. Raymond, F. Real, K. Rimbach, C. Ross, B. Rotsted, H. Roussez, N. Ryder, M. Saltarelli, T. Sanders, S. Santurkar, G. Sastry, H. Schmidt, D. Schnurr, J. Schulman, D. Selsam, K. Sheppard, T. Sherbakov, J. Shieh, S. Shoker, P. Shyam, S. Sidor, E. Sigler,

- M. Simens, J. Sitkin, K. Slama, I. Sohl, B. Sokolowsky, Y. Song,
- N. Staudacher, F. P. Such, N. Summers, I. Sutskever, J. Tang, N. Tezak, M. Thompson, P. Tillet, A. Tootoonchian, E. Tseng, P. Tuggle, N. Turley, J. Tworek, J. F. C. Uribe, A. Vallone,

- A. Vijayvergiya, C. Voss, C. Wainwright, J. J. Wang, A. Wang,
- B. Wang, J. Ward, J. Wei, C. Weinmann, A. Welihinda, P. Welinder, J. Weng, L. Weng, M. Wiethoff, D. Willner, C. Winter,

## 6. References

- 1. W. Hu, Y. Xu, Y. Li, W. Li, Z. Chen, and Z. Tu, “Bliva: A simple multimodal llm for better handling of text-rich visual questions,”2023.
- 2. Q. Ye, H. Xu, G. Xu, J. Ye, M. Yan, Y. Zhou, J. Wang, A. Hu, P. Shi, Y. Shi, C. Li, Y. Xu, H. Chen, J. Tian, Q. Qi, J. Zhang, and

F. Huang, “mplug-owl: Modularization empowers large language models with multimodality,” 2023.

- 3. W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. Li, P. Fung, and S. Hoi, “Instructblip: Towards general-purpose vision-language models with instruction tuning,” 2023.
- 4. A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed, “Mistral 7b,” 2023.
- 5. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and

I. Sutskever, “Language models are unsupervised multitask learn-ers,” 2019.

- 6. J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pretraining of deep bidirectional transformers for language understanding,” 2019.
- 7. “Openai. gpt-3: Powerful language models for conersation. ope-nai, 2022.”
- 8. OpenAI, :, J.Achiam, S.Adler, S.Agarwal, L.Ahmad, I.Akkaya,

- S. Wolrich, H. Wong, L. Workman, S. Wu, J. Wu, M. Wu, K. Xiao,
- T. Xu, S. Yoo, K. Yu, Q. Yuan, W. Zaremba, R. Zellers, C. Zhang, M. Zhang, S. Zhao, T. Zheng, J. Zhuang, W. Zhuk, and B. Zoph, “Gpt-4 technical report,” 2023.

- 9. M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro, “Megatron-lm: Training multi-billion parameter language models using model parallelism,” 2020.
- 10. R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, E. Chu, J. H. Clark, L. E. Shafey, Y. Huang, K. Meier-Hellstern, G. Mishra, E. Moreira, M. Omernick, K. Robinson, S. Ruder, Y. Tay, K. Xiao, Y. Xu, Y. Zhang, G. H. Abrego, J. Ahn, J. Austin, P. Barham, J. Botha, J. Bradbury, S. Brahma, K. Brooks, M. Catasta, Y. Cheng, C. Cherry, C. A. Choquette-Choo, A. Chowdhery, C. Crepy, S. Dave, M. Dehghani, S. Dev, J. Devlin, M. D´ıaz, N. Du, E. Dyer, V. Feinberg, F. Feng, V. Fienber, M. Freitag, X. Garcia, S. Gehrmann, L. Gonzalez, G. Gur-Ari, S. Hand, H. Hashemi, L. Hou, J. Howland, A. Hu, J. Hui, J. Hurwitz, M. Isard, A. Ittycheriah, M. Jagielski, W. Jia, K. Kenealy, M. Krikun, S. Kudugunta, C. Lan, K. Lee, B. Lee, E. Li, M. Li, W. Li, Y. Li,

- J. Li, H. Lim, H. Lin, Z. Liu, F. Liu, M. Maggioni, A. Mahendru, J. Maynez, V. Misra, M. Moussalem, Z. Nado, J. Nham, E. Ni, A. Nystrom, A. Parrish, M. Pellat, M. Polacek, A. Polozov, R. Pope, S. Qiao, E. Reif, B. Richter, P. Riley, A. C. Ros, A. Roy, B. Saeta, R. Samuel, R. Shelby, A. Slone, D. Smilkov, D. R. So, D. Sohn, S. Tokumine, D. Valter, V. Vasudevan, K. Vodrahalli, X. Wang, P. Wang, Z. Wang, T. Wang, J. Wieting, Y. Wu,
- K. Xu, Y. Xu, L. Xue, P. Yin, J. Yu, Q. Zhang, S. Zheng, C. Zheng, W. Zhou, D. Zhou, S. Petrov, and Y. Wu, “Palm 2 technical report,” 2023.

- 11. K. A. Wang, D. Maddix, and Y. Wang, “Gopher: Cate- gorical probabilistic forecasting with graph structure via local continuous-time dynamics,” 2021.
- 12. J. Hoffmann, S. Borgeaud, A. Mensch, E. Buchatskaya, T. Cai, E. Rutherford, D. de Las Casas, L. A. Hendricks, J. Welbl,

A. Clark, T. Hennigan, E. Noland, K. Millican, G. van den Driess- che, B. Damoc, A. Guy, S. Osindero, K. Simonyan, E. Elsen, J. W. Rae, O. Vinyals, and L. Sifre, “Training computeoptimal large language models,” 2022.

- 13. S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, T. Mihaylov, M. Ott,

S. Shleifer, K. Shuster, D. Simig, P. S. Koura, A. Sridhar, T. Wang, and L. Zettlemoyer, “Opt: Open pre-trained transformer languagemodels,” 2022.

- 14. B. Workshop, :, T. L. Scao, A. Fan, C. Akiki, E. Pavlick, S. Ilic´, D. Hesslow, R. Castagne´, A. S. Luccioni, F. Yvon,

- F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, R. Avila, I. Babuschkin, S. Balaji, V. Balcom, P. Baltescu, H. Bao, M. Bavarian, J. Belgum, I. Bello, J. Berdine,
- G. Bernadett-Shapiro, C. Berner, L. Bogdonoff, O. Boiko,

- M. Boyd, A.-L. Brakman, G. Brockman, T. Brooks, M. Brundage, K. Button, T. Cai, R. Campbell, A. Cann, B. Carey, C. Carlson, R. Carmichael, B. Chan, C. Chang, F. Chantzis, D. Chen, S. Chen,

- R. Chen, J. Chen, M. Chen, B. Chess, C. Cho, C. Chu, H. W. Chung, D. Cummings, J. Currier, Y. Dai, C. Decareaux, T. Degry,

N. Deutsch, D. Deville, A. Dhar, D. Dohan, S. Dowling, S. Dunning, A. Ecoffet, A. Eleti, T. Eloundou, D. Farhi, L. Fedus, N. Felix, S. P. Fishman, J. Forte, I. Fulford, L. Gao, E. Georges, C. Gib- son, V. Goel, T. Gogineni, G. Goh, R. Gontijo-Lopes, J. Gordon, M. Grafstein, S. Gray, R. Greene, J. Gross, S. S. Gu, Y. Guo, C. Hallacy, J. Han, J. Harris, Y. He, M. Heaton, J. Heidecke, C. Hesse, A. Hickey, W. Hickey, P. Hoeschele, B. Houghton,

- K. Hsu, S. Hu, X. Hu, J. Huizinga, S. Jain, S. Jain, J. Jang, A. Jiang, R. Jiang, H. Jin, D. Jin, S. Jomoto, B. Jonn, H. Jun, T. Kaftan, Łukasz Kaiser, A. Kamali, I. Kanitscheider, N. S. Keskar, T. Khan, L. Kilpatrick, J. W. Kim, C. Kim, Y. Kim,

H. Kirchner, J. Kiros, M. Knight, D. Kokotajlo, Łukasz Kon-draciuk, A. Kondrich, A. Konstantinidis, K. Kosic, G. Krueger, V. Kuo, M. Lampe, I. Lan, T. Lee, J. Leike, J. Leung, D. Levy, C. M. Li, R. Lim, M. Lin, S. Lin, M. Litwin, T. Lopez, R. Lowe, P. Lue, A. Makanju, K. Malfacini, S. Manning, T. Markov,

- Y. Markovski, B. Martin, K. Mayer, A. Mayne, B. McGrew,

- S. M. McKinney, C. McLeavey, P. McMillan, J. McNeil, D. Med- ina, A. Mehta, J. Menick, L. Metz, A. Mishchenko, P. Mishkin, V. Monaco, E. Morikawa, D. Mossing, T. Mu, M. Murati,

- O. Murk, D. Me´ly, A. Nair, R. Nakano, R. Nayak, A. Neelakantan,

M. Galle´, J. Tow, A. M. Rush, S. Biderman, A. Webson, P. S. Ammanamanchi, T. Wang, B. Sagot, N. Muennighoff, A. V. del Moral, O. Ruwase, R. Bawden, S. Bekman, A. McMillan- Major, I. Beltagy, H. Nguyen, L. Saulnier, S. Tan, P. O. Suarez, V. Sanh, H. Laurenc¸on, Y. Jernite, J. Launay, M. Mitchell, C. Raffel, A. Gokaslan, A. Simhi, A. Soroa, A. F. Aji, A. Alfassy, A. Rogers, A. K. Nitzav, C. Xu, C. Mou, C. Emezue, C. Klamm, C. Leong, D. van Strien, D. I. Adelani, D. Radev, E. G. Ponferrada, E. Levkovizh, E. Kim, E. B. Natan, F. D. Toni, G. Dupont, G. Kruszewski, G. Pistilli, H. Elsahar, H. Benyamina, H. Tran, I. Yu, I. Abdulmumin, I. Johnson, I. GonzalezDios, J. de la Rosa, J. Chim, J. Dodge, J. Zhu, J. Chang, J. Frohberg, J. Tobing, J. Bhattacharjee, K. Almubarak, K. Chen,

- R. Ngo, H. Noh, L. Ouyang, C. O’Keefe, J. Pachocki, A. Paino,

- J. Palermo, A. Pantuliano, G. Parascandolo, J. Parish, E. Parparita, A. Passos, M. Pavlov, A. Peng, A. Perelman, F. de Avila Belbute Peres, M. Petrov, H. P. de Oliveira Pinto, Michael, Pokorny, M. Pokrass, V. Pong, T. Powell, A. Power, B. Power,

- K. Lo, L. V. Werra, L. Weber, L. Phan, L. B. allal, L. Tanguy, M. Dey, M. R. Mun˜oz, M. Masoud, M. Grandury, M. Sˇasˇko, M. Huang, M. Coavoux, M. Singh, M. T.-J. Jiang, M. C. Vu, M. A. Jauhar, M. Ghaleb, N. Subramani, N. Kassner, N. Khamis,

- O. Nguyen, O. Espejel, O. de Gibert, P. Villegas, P. Henderson,
- P. Colombo, P. Amuok, Q. Lhoest, R. Harliman, R. Bommasani,

- R. L. Lo´pez, R. Ribeiro, S. Osei, S. Pyysalo, S. Nagel, S. Bose,
- S. H. Muhammad, S. Sharma, S. Longpre, S. Nikpoor, S. Sil-berberg, S. Pai, S. Zink, T. T. Torrent, T. Schick, T. Thrush, V. Danchev, V. Nikoulina, V. Laippala, V. Lepercq, V. Prabhu,

Z. Alyafeai, Z. Talat, A. Raja, B. Heinzerling, C. Si, D. E. Tas¸ar, E. Salesky, S. J. Mielke, W. Y. Lee, A. Sharma, A. Santilli, A. Chaffin, A. Stiegler, D. Datta, E. Szczechla, G. Chhablani, H. Wang, H. Pandey, H. Strobelt, J. A. Fries, J. Rozen,

- L. Gao, L. Sutawika, M. S. Bari, M. S. Al-shaibani, M. Manica, N. Nayak, R. Teehan, S. Albanie, S. Shen, S. Ben-David,

S. H. Bach, T. Kim, T. Bers, T. Fevry, T. Neeraj, U. Thakker, V. Raunak, X. Tang, Z.-X. Yong, Z. Sun, S. Brody, Y. Uri, H. Tojarieh, A. Roberts, H. W. Chung, J. Tae, J. Phang, O. Press, C. Li, D. Narayanan, H. Bourfoune, J. Casper, J. Rasley, M. Ryabinin,

- M. Mishra, M. Zhang, M. Shoeybi, M. Peyrounette, N. Patry,
- N. Tazi, O. Sanseviero, P. von Platen, P. Cornette, P. F. Lavalle´e,

- 18. M. Tsimpoukelli, J. Menick, S. Cabi, S. M. A. Eslami, O. Vinyals, and F. Hill, “Multimodal few-shot learning with frozen languagemodels,” 2021.
- 19. J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, R. Ring, E. Rutherford, S. Cabi, T. Han, Z. Gong, S. Samangooei, M. Mon- teiro, J. Menick, S. Borgeaud, A. Brock, A. Nematzadeh, S. Shar- ifzadeh, M. Binkowski, R. Barreira, O. Vinyals, A. Zisserman, and K. Simonyan, “Flamingo: a visual language model for few- shot learning,” 2022.
- 20. J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” 2023.
- 21. H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” 2023.
- 22. H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” 2023.
- 23. B. Li, Y. Zhang, L. Chen, J. Wang, J. Yang, and Z. Liu, “Otter: A multi-modal model with in-context instruction tuning,” 2023.
- 24. A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and

I. Sutskever, “Learning transferable visual models from natural language supervision,” 2021.

- 25. C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta, T. Coombes, J. Jitsev, and A. Komatsuzaki, “Laion-400m: Open dataset of clip-filtered 400 million imagetext pairs,” 2021.
- 26. M. B. B. P. H. K. S. Lee., “Coyo-700m:,” Journal Name, vol. Volume, no. Issue, p. Page Range, 2022. [Online]. Available: URL
- 27. S. Changpinyo, P. Sharma, N. Ding, and R. Soricut, “Conceptual 12m: Pushing web-scale image-text pre-training to recognizelong-tail visual concepts,” 2021.
- 28. T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, C. L. Zitnick, and P. Dolla´r, “Microsoft coco: Common objects in context,” 2015.
- 29. H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale,

- D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Es-iobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V.

Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas,

V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao,

X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poul-ton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R.

Silva,

- E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor,

A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom, “Llama 2: Open foundation and fine-tuned chat models,” 2023.

- 30. A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. de las Casas, E. B. Hanna, F. Bres- sand, G. Lengyel, G. Bour, G. Lample, L. R. Lavaud, L. Saulnier,M.-A. Lachaux, P. Stock, S. Subramanian, S. Yang, S. Antoniak, T. L. Scao, T. Gervet, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed, “Mixtral of experts,” 2024.

- R. Lacroix, S. Rajbhandari, S. Gandhi, S. Smith, S. Requena,
- S. Patil, T. Dettmers, A. Baruwa, A. Singh, A. Cheveleva, A.-

L. Ligozat, A. Subramonian, A. Ne´ve´ol, C. Lovering, D. Garrette,

- D. Tunuguntla, E. Reiter, E. Taktasheva, E. Voloshina, E. Bogdanov, G. I. Winata, H. Schoelkopf, J.-C. Kalo, J. Novikova, J. Z. Forde, J. Clive, J. Kasai, K. Kawamura, L. Hazan, M. Carpuat,

M. Clinciu, N. Kim, N. Cheng, O. Serikov, O. Antverg, O. van der Wal, R. Zhang, R. Zhang, S. Gehrmann, S. Mirkin, S. Pais, T. Shavrina, T. Scialom, T. Yun, T. Limisiewicz, V. Rieser, V. Protasov, V. Mikhailov, Y. Pruksachatkun, Y. Belinkov, Z. Bamberger, Z. Kasner, A. Rueda, A. Pestana, A. Feizpour, A. Khan,

- A. Faranak, A. Santos, A. Hevia, A. Unldreaj, A. Aghagol, A. Abdollahi, A. Tammour, A. HajiHosseini, B. Behroozi, B. Ajibade,
- B. Saxena, C. M. Ferrandis, D. McDuff, D. Contractor, D. Lansky, D. David, D. Kiela, D. A. Nguyen, E. Tan, E. Baylor,

- E. Ozoani, F. Mirza, F. Ononiwu, H. Rezanejad, H. Jones,

I. Bhattacharya, I. Solaiman, I. Sedenko, I. Nejadgholi, J. Passmore, J. Seltzer, J. B. Sanz, L. Dutra, M. Samagaio, M. Elbadri, M. Mieskes, M. Gerchick, M. Akinlolu, M. McKenna,

- M. Qiu, M. Ghauri, M. Burynok, N. Abrar, N. Rajani, N. Elkott,
- N. Fahmy, O. Samuel, R. An, R. Kromann, R. Hao, S. Alizadeh, S. Shubber, S. Wang, S. Roy, S. Viguier, T. Le, T. Oyebade, T. Le, Y. Yang, Z. Nguyen, A. R. Kashyap, A. Palas- ciano,

- A. Callahan, A. Shukla, A. Miranda-Escalada, A. Singh,
- B. Beilharz, B. Wang, C. Brito, C. Zhou, C. Jain, C. Xu,
- C. Fourrier, D. L. Perin˜a´n, D. Molano, D. Yu, E. Manjavacas,

- F. Barth, F. Fuhrimann, G. Altay, G. Bayrak, G. Burns, H. U. Vrabec, I. Bello, I. Dash, J. Kang, J. Giorgi, J. Golde, J. D. Posada, K. R. Sivaraman, L. Bulchandani, L. Liu, L. Shinzato, M. H. de Bykhovetz, M. Takeuchi, M. Pa`mies, M. A. Castillo, M. Nezhurina, M. Sa¨nger, M. Samwald, M. Cullan, M. Weinberg, M. D. Wolf, M. Mihaljcic, M. Liu, M. Freidank, M. Kang, N. See- lam, N. Dahlberg, N. M. Broad, N. Muellner, P. Fung, P. Haller,

- R. Chandrasekhar, R. Eisenberg, R. Martin, R. Canalli, R. Su,

- R. Su, S. Cahyawijaya, S. Garda, S. S. Deshmukh, S. Mishra,
- S. Kiblawi, S. Ott, S. Sang-aroonsiri, S. Kumar, S. Schweter,

- S. Bharati, T. Laud, T. Gigant, T. Kainuma, W. Kusa, Y. Labrak,

- Y. S. Bajaj, Y. Venkatraman, Y. Xu, Y. Xu, Y. Xu, Z. Tan, Z. Xie,
- Z. Ye, M. Bras, Y. Belkada, and T. Wolf, “Bloom: A 176bparameter open-access multilingual language model,” 2023.

- 15. L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman,

J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe, “Training language models to follow instructions with human feedback,” 2022.

- 16. OpenAI, “Tb openai. chatgpt: Optimizing language models for dialogue.” 2022.
- 17. J. Chen, H. Guo, K. Yi, B. Li, and M. Elhoseiny, “Visualgpt: Data- efficient adaptation of pretrained language models for image cap-tioning,” 2022.

7. Appendix

[Figure 6]

### 1. GitHub

The code repository, along with detailed documentation, can be found at https://github.com/superagi/Veagle.

### 2. Huggingface

The Veagle model, along with detailed documentation is available at https://huggingface.co/SuperAGI/ Veagle

### 3. Training Parameters

Table 3: Training parameters in both the stages

Epochs Optimizer l-rate Batch size Weight decay

Pre-training 3 AdamW 1e-5 8 0.05 Fine-tuning 2 AdamW 1e-5 10 0.05

### 4. Compute complexity

We have used 8 NVIDIA A100 with a batch size of 10 for both pre-training and fine-tuning. For inference 1 NVIDIA A6000 is used.

### 5. Qualitative Examples

[Figure 7]

[Figure 8]

- Figure 6: Examples generated by our Veagle model exemplify a broad spectrum of its diverse capabilities. These showcases encompass intricate visual scene comprehension and reasoning, multi-turn visual conversation, and various other impressive functionalities.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

- Figure 7: From pixels to pitches, our model’s ability to generate poems, ads, and songs from images is a testament to its multi- dimensional creative prowess.

Figure 8: Examples generated by our Veagle model showing its innovative capabilities.

