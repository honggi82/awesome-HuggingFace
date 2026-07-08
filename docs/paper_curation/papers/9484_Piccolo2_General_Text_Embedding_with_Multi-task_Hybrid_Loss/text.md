arXiv:2405.06932v1[cs.CL]11May2024

Piccolo2: General Text Embedding with Multi-task Hybrid Loss Training

Junqin Huang, Zhongjie Hu, Zihao Jing, Mengya Gao, Yichao Wu SenseTime Research

Abstract

In this report, we introduce Piccolo2, an embedding model that surpasses other models in the comprehensive evaluation over 6 tasks on CMTEB [36] benchmark, setting a new state-of-the-art. Piccolo2 primarily leverages an eﬃcient multi-task hybrid loss training approach, eﬀectively harnessing textual data and labels from diverse downstream tasks. In addition, Piccolo2 scales up the embedding dimension and uses MRL [14] training to support more ﬂexible vector dimensions. The latest information of piccolo models can be accessed via: piccolo

# 1 Introduction

Text embedding models play a pivotal role in natural language processing and machine learning. By encoding texts into structured numerical representations, known as text embeddings, these models encapsulate semantic and contextual information of words, phrases, or entire documents within a dense, lowdimensional vector space [27]. Such embeddings are indispensable for various downstream NLP tasks, including classiﬁcation, clustering, retrieval, and sentence similarity.

Contrastive learning stands out as the most eﬀective technique for training text embedding models [6]. It presents text semantic representations by minimizing the distance between positive pairs and maximizing the distance between negative pairs. Beyond its application in natural language processing (NLP), contrastive learning also proves pivotal in visual [8] [5] and multi-modal [25] representation learning. Recent advanced text embedding works [36] [33] [18] primarily rely on a two-stage pretrain-ﬁnetune pipeline to acquire general text embedding models. Pre-training utilizes weakly supervised data sourced from large-scale crawling eﬀorts, while ﬁne-tuning involves reﬁning the model with high-quality text pairs obtained through data mining or manual annotation.

In this paper, we introduce Piccolo2 and propose a multi-task hybrid training method to better utilize textual data and labels from diﬀerent granularities (e.g. usually the labels of STS [30] tasks are more ﬁne-grained than retrieval [37] tasks). In addition to training methods, high-quality data has long been recognized as an important component in embedding training [33] [15]. Piccolo2 also devise a data synthetic framework and a hard negative mining approach to continually augment high-quality datasets. We demonstrate our training approach in Section 2, the composition and source of the data in Section 3, and our experimental results in Section 4.

# 2 Training Details

- 2.1 Multi-Task Hybrid Loss Training

Previous training processes for embedding models are mostly relied on the standard InfoNCE loss [7] with in-batch negative samples, which typically achieves robust representations through the utilization of large batch size and a great number of negative samples. However, standard InfoNCE cannot resolve all situations in the present landscape of downstream tasks for embedding models. For example, STS tasks usually achieve inferior results by training with InfoNCE loss. Furthermore, classiﬁcation tasks and clustering tasks have also not been utilized in the training of general embedding models. Hence, we employ a multi-task hybrid loss training method to Piccolo2, which utilize various training loss for diﬀerent downstream task and prove its superior performance. In the following parts, we introduce the loss functions of our hybrid training methods in detail.

- 2.1.1 Retrieval and Reranking Loss

For retrieval and reranking tasks [37], we follow previous works [33] [18] and use the standard InfoNCE loss with in-batch negative.

Lre = −

1 n i

[Figure 1]

log

es(q,d+)/τ es(q,d+)/τ + j es(q,d−)/τ (1)

[Figure 2]

where s(q,d) is a scoring function between query q and passage p, often deﬁned as the cosine similarity and τ is the scale temperature.

- 2.1.2 STS and PairClassiﬁcation Loss

Most previous studies [33] [36] also use InfoNCE loss to optimize tasks on natural language inference (NLI) datasets, where contradictory sentences are usually treated as hard negatives, while entailment or neutral sentences are treated as

positives. However, since the original STS and pair-classiﬁcation tasks usually have more ﬁne-grained labels (such as score values), converting them into triplets inevitably leads to information loss, consequently yielding inferior results. Therefore, we directly employ the cosent loss function [32], a ranking loss function speciﬁcally designed for the text pairs with ﬁne-grained labels.

 1 +

Lsts = log

s(xi,xj)>s(xm,xn)

  (2)

cos(xm,xn)−cos(xi,xj) τ

e

[Figure 3]

where τ is the scale temperature, cos(·) is the cosine similarity function, and s(u, v) is the similarity between u and v.

- 2.1.3 Classiﬁcation and Clustering Loss

For classiﬁcation and clustering tasks, we leverage the SFR embedding method [28] to seamlessly integrate data into training process. To be more speciﬁc, we reformat classiﬁcation and clustering datasets into contrastive triplets. For example, in a classiﬁcation task with 10 categories, each input text x is treated with its target label y+ as the positive pair, while the remaining 9 labels y− serve as negative pairs. This approach is similarly applied to clustering tasks. This diﬀers from the approach mentioned in [15], where documents sharing the same label are considered as positive pairs, and those with diﬀerent labels are considered as negative pairs. In contrast, pairing documents with their corresponding labels is more intuitive.

Lcls = −

1 n i

[Figure 4]

log

es(x,y+)/τ

[Figure 5]

es(x,y+)/τ + j es(x,y−)/τ (3) Formula 1 and Formula 3 share the same structure. However, it’s important to stress that, for Lcls, in-batch negatives are no longer used due to the fact that it can easily lead to conﬂict training targets. As a result, only label negatives are used in Lcls.

- 2.1.4 Multi-Task Hybrid Loss

During training, we blend the aforementioned three loss functions and ﬁnal loss function is formulated as follows:

 

Lcls if task is classiﬁcation or clustering, Lsts if task is sts or pair-classiﬁcation, Lre if task is retrieval or reranking,

L =



(4)

- 2.2 Dimension Scaling up and MRL Training

- 2.2.1 Dimension Scaling up

Inspired by OpenAI’s text-embedding-v3 [23] we scale up the embedding vector dimension from 768 to 1792 to increase the model capacity, compared with

- Piccolo1 [12]. During the ﬁne-tuning stage, We directly add a learnable linear layer to the last layer of BERT, with a size of (origin dim,scale dim). The operation can be formulated as:

[Figure 6]

[Figure 7]

emb = Linear(pool(BERT(text))) (5)

2.2.2 MRL Training

Matryoshka Representation Learning [14] introduces a novel approach for training embedding models with ﬂexible dimension lengths, ensuring the utility of embeddings even after dimensionality reduction. This technique not only enhances the speed of processing but also signiﬁcantly reduces storage requirements with slight performance drop. This has proven to be very eﬀective in text-embedding v3 released by OpenAI [23] and has been adopted by many advanced embeddings works such as [29] [15] [10]. In the training of Piccolo2, we also employ the MRL approach and validate its eﬀectiveness.

3 Datasets

- Piccolo2 is trained on a diverse range of tasks. We collect open-source data and employ a helpful data synthetic pipeline to generate more training samples. Hard negative mining is also applied on dataset for retrieval tasks.

- 3.1 Datasets Synthetic Pipeline

Recently, a lot of work has focused on generating large-scale sample pairs through GPT4 to eliminate the need for complicated manual annotation [34] [15]. This is a very useful method, especially in some special scenarios where data is scarce. We follow previous work [34] by creating approximately 200k retrieval datasets and incorporating them into our training datasets. For further details, please refer to the Appendix.

- 3.2 Dataset Details

For retrieval tasks, we collect data from MMarcoRetrieval [4], T2Rerieval [37], DuRetrieval [24], CmedqaRetrieval [39],CovidRetrieval [35], and Multi-CPR [19]. For MMarcoRetrieval, 400k subsets are sampled. Clustering tasks consist of data from Thunews [16] and CSL [17], with ﬁlters applied to exclude

development and test sets in the CMTEB clustering framework. In classiﬁcation, our training dataset utilize datasets from Tnews, Iﬂytek [38], Multilingualsentiments [21], JDReview [13], OnlineShopping, and Waimai [31]. For Semantic Textual Similarity (STS) and Pair Classiﬁcation, training data includes NLIzh [30], Afqmc, Qbqtc, Cmnli [38] and Ocnli [9]. Additionally, about 200k short to long retrieval task data are generated through the data synthesis pipeline mentioned in Section 3.1. For most of the data sets mentioned here, we use its train split to prevent overlap with the CMTEB test set. We list the composition of the dataset in Table 1.

[Figure 8]

Task training data meta # Sampled STS text, text pair, score 730k Pair Classiﬁcation text, text pair, score 440k Retrieval text, pos text, neg text 1.1M Retrieval generated text, pos text, neg text 200k Clustering text, pos label, neg label 1M Classifcation text, pos label, neg label 220k Total ∼3.7M

[Figure 9]

[Figure 10]

Table 1: Data mixture for supervised ﬁne-tuning.

- 3.3 Hard Negative Mining

For each retrieval task, we use piccolo-base-zh [12] to conduct negative sample mining. We randomly select 15 samples from the mining negatives of rank 50 - 100 as the ﬁnal hard negative samples. We avoid using higher-rank negative samples as their inclusion typically leads to a decline in performance. This is caused by a variety of reasons, such as inaccurate dataset annotation.

- 4 Experiments

- 4.1 Implementation details

During the training phase, we employ the AdamW [20] optimizer with an initial learning rate of 1e-5, incorporating cosine decay. The batch size per GPU is set to 256 with only 1 negative sample for each retrieval data point. The model undergoes training for 2500 steps with a maximum input length limited to 512, and for simplicity we do not include any instructions. For MRL training, we conﬁgure the matryoshka representation dimension as 256, 384, 768, 1024, 1536 and 1972. For eﬃciency, we use mixed precision training, gradient checkpointing and deepspeed ZERO-stage1 [26]. It ﬁnally takes 32 A100 GPU and 6 hours for the whole ﬁne-tuning process. It is worth noting that since diﬀerent tasks (e.g. clustering, retrieval) usually have diﬀerent number of negatives, we do not use cross device negative because this would be very troublesome to implement in the context of multi loss ﬁne-tuning we mentioned in section 2.1.

[Figure 11]

[Figure 12]

[Figure 13]

Dim. # Params. Class. Cluster. Pair. Rerank. Retr. STS Avg.

[Figure 14]

[Figure 15]

[Figure 16]

gte-Qwen1.5-7B-instruct [1] 4,096 7B 73.35 67.08 88.52 66.38 70.62 62.32 69.56 acge-text-embedding [11] 1,792 300M 72.75 58.7 87.84 67.98 72.93 62.09 69.07 aliyun-text-embedding [2] 1,792 n/a 71.74 53.75 88.1 68.27 74.41 62.46 68.71 stella-mrl-large [10] 1,792 300M 71.56 54.32 88.08 68.45 73.52 62.48 68.55 baichuan-text-embedding [3] 1,024 n/a 72.84 56.88 82.32 69.67 73.12 60.07 68.34

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

piccolo2 1,792 300M 74.59 62.17 90.24 70 74.36 63.5 70.95

[Figure 28]

- Table 2: Results on CMTEB. We report the average performance on six diﬀerent tasks: Classiﬁcation (Class.), Clustering (Cluter.), Pair Classiﬁcation (Pair.), Reranking (Rerank.), Retrieval (Retr.), and STS. The last column shows the average performance across all datasets from the six tasks.

[Figure 29]

[Figure 30]

Class. Cluster. Pair. Rerank. Retrieval STS Avg.

[Figure 31]

[Figure 32]

La : Lre 72.85 54.72 86.14 69.71 74.05 61.88 68.75 Lb : Lre + Lsts 73.10 53.39 90.02 69.08 74.25 63.35 69.87 Lc : Lre + Lsts + Lcls 74.59 62.17 90.24 70 74.36 63.5 70.95

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

- Table 3: We contrast the eﬀects of employing the hybrid loss against utilizing the InfoNCE loss. The adoption of the hybrid loss has yielded substantial enhancements across clustering, classiﬁcation, semantic text similarity (STS), and pair classiﬁcation tasks.

- 4.2 CMTEB evaluation

MTEB (Massive Text Embedding Benchmark) [22] is a widely used benchmark for evaluating large-scale text embedding tasks. Recently, Xiao et al. [36] has introduced CMTEB, which builds upon MTEB by incorporating numerous Chinese datasets, thereby greatly enhancing the evaluation capabilities for Chinese language models. In this report, we conduct experiments on CMTEB, which has 31 datasets spanning across 6 categories: Classiﬁcation, Clustering, Pair Classiﬁcation, Rerank, Retrieval and STS. Table 2 presents a comparison of our models with others on the CMTEB benchmark. Notably, our best-performing Piccolo2 model surpasses the previous state-of-the-art BERT-based model, acgetext-embeddings, by 1.9 points.

- 4.3 Ablation Study

- 4.3.1 Multi-Task Hybrid Loss tuning

To demonstrate the eﬀectiveness of hybrid loss tuning compared to directly using InfoNCE for ﬁne-tuning, we conduct a simple ablation experiment. In this experiment, we design three diﬀerent loss function variants and use the same dataset for training. Our three loss function variants are:

- • Loss Variant A (La): Only InfoNCE is used for training on the six tasks.
- • Loss Variant B (Lb): Cosent loss is used speciﬁcally for the STS and Pair Classiﬁcation tasks, while InfoNCE is used for the rest Retrieval,

[Figure 40]

[Figure 41]

[Figure 42]

Eval Dim. Class. Cluster. Pair. Rerank. Retrieval STS Avg. 1,792 74.59 62.17 90.24 70 74.36 63.5 70.95 1,536 74.46 62.67 90.28 69.96 74.35 63.5 70.97 1,280 74.29 62.39 90.27 69.8 74.29 63.51 70.87 1,024 74.05 62.27 90.3 69.93 74.27 63.52 70.8 768 73.79 62.29 90.3 69.62 74.21 63.52 70.69 512 73.36 61.85 90.17 69.37 73.87 63.47 70.41 256 72.74 62.24 90.1 68.89 72.88 63.4 69.99

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

[Figure 57]

[Figure 58]

- Table 4: Results on MRL evaluation. We train the model using a 1792dimensional embedding space and evaluat its performance across various dimensions. Notably, even when the embedding dimension is signiﬁcantly reduced by roughly eightfold (from 1792 to 256), the model’s performance degradation is minimal, at only around 1 point.

Classiﬁcation and Clustering tasks.

• Loss Variant C (Lc): Based on Lb, it specially utilizes loss function 3 for both Clustering and Classiﬁcation tasks. Lc is also our ﬁnal hybrid loss function 4.

As presented in Table 3, Lb outperforms La notably in both pair classiﬁcation and STS tasks. This ﬁnding is consistent with previous work [32], which claims that rank loss is often a superior choice for pairs with ﬁne-grained labels. Finally, after reformatting clustering and classiﬁcation datasets as contrastive triplets as we discussed in 2.1.3, Lc achieves substantial improvements in both classiﬁcation and clustering tasks.

- 4.3.2 Large dimension

We also compare the impact of training embedding dimension on the ﬁnal performance. The experimental results are presented in Table 5. Contrary to our expectations, expanding the dimension does not yield additional beneﬁts to the embedding model.

- 4.3.3 MRL training

We also examine the impact of employing MRL on the model’s performance. The results presented in Table 6 indicate that whether we utilize MRL or not, there is minimal alteration in the model’s performance. Yet it does demonstrate the eﬀectiveness of MRL as it enables the support of ﬂexible dimension length without sacriﬁcing performance compared to single-dimensional training.

[Figure 59]

[Figure 60]

[Figure 61]

Training Dim. Class. Cluster. Pair. Rerank. Retrieval STS Avg. 1,024 74.42 62.2 90.02 69.86 74.21 63.34 70.79 1,792 74.59 62.17 90.24 70 74.36 63.5 70.95 3,072 74.49 62.22 90.42 69.83 74.3 63.42 70.87

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Table 5: Model’s performance under diﬀerent training embedding dimension.

[Figure 70]

[Figure 71]

[Figure 72]

Eval Dim Class. Cluster. Pair. Rerank. Retrieval STS Avg.

[Figure 73]

[Figure 74]

[Figure 75]

w/ MRL 1,792 74.59 62.17 90.24 70 74.36 63.5 70.95 w/o MRL 1,792 74.5 62.33 90.13 69.75 74.39 63.38 70.84

[Figure 76]

[Figure 77]

[Figure 78]

Table 6: The impact of using MRL training.

# 5 Conclusion

In this report, we present Piccolo2, the new state-of-the-art chinese embedding model. Piccolo2 mainly focuses on general downstream training by leveraging the eﬀectiveness of multi-task hybrid loss. In addition, Piccolo2 also supports the use of vectors with ﬂexible dimension lengths through MRL training. The latest information about the Piccolo2 model will be synchronized on Hugging Face: https://huggingface.co/sensenova

# References

- [1] alinlp. gte-qwen1.5-7b-instruct. hugging face, 2024. URL: https://huggingface.co/Alibaba-NLP/gte-Qwen1.5-7B-instruct.
- [2] aliyun. aliyun-text-embedding. hugging face, 2024. URL: https://help.aliyun.com/zh/open-search.
- [3] baichuan. baichuan-embedding, 2024. URL: https://platform.baichuan-ai.com/docs/text-Embedding.
- [4] Luiz Bonifacio, Vitor Jeronymo, Hugo Queiroz Abonizio, Israel Campiotti, Marzieh Fadaee, Roberto Lotufo, and Rodrigo Nogueira. mmarco: A multilingual version of the ms marco passage ranking dataset. arXiv preprint arXiv:2108.13897, 2021.

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

- [5] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoﬀrey Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PMLR,

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

- 2020.

[6] Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910,

[Figure 92]

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

- 2021.

- [7] Michael Gutmann and Aapo Hyv¨rinen. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of the thirteenth international conference on artiﬁcial intelligence and statistics, pages 297–304. JMLR Workshop and Conference Proceedings, 2010.

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

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

- [8] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

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

- [9] Hai Hu, Kyle Richardson, Liang Xu, Lu Li, Sandra K¨ubler, and Lawrence S Moss. Ocnli: Original chinese natural language inference. arXiv preprint arXiv:2010.05444, 2020.

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

- [10] infgrad. stella-mrl-large-zh-v3.5-1792d. hugging face, 2024. URL: https://huggingface.co/infgrad/stella-mrl-large-zh-v3.5-1792d.
- [11] intsig. acge-text-embedding. hugging face, 2024. URL: https://huggingface.co/aspire/acge_text_embedding.
- [12] junqin huang. piccolo-large-zh. hugging face, 2023. URL: https://huggingface.co/sensenova/piccolo-large-zh.
- [13] kuroneko5943. Jdreview dataset, 2023. URL: https://huggingface.co/datasets/kuroneko5943/jd21.
- [14] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, et al. Matryoshka representation learning. Advances in Neural Information Processing Systems,35:30233–30249, 2022.

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

- [15] Jinhyuk Lee, Zhuyun Dai, Xiaoqi Ren, Blair Chen, Daniel Cer, Jeremy R Cole, Kai Hui, Michael Boratko, Rajvi Kapadia, Wen Ding, et al. Gecko: Versatile text embeddings distilled from large language models. arXiv preprint arXiv:2403.20327, 2024.

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

- [16] Jingyang Li, Maosong Sun, and Xian Zhang. A comparison and semiquantitative analysis of words and character-bigrams as features in chinese text categorization. In proceedings of the 21st international conference on computational linguistics and 44th annual meeting of the association for computational linguistics, pages 545–552, 2006.

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

- [17] Yudong Li, Yuqing Zhang, Zhe Zhao, Linlin Shen, Weijie Liu, Weiquan Mao, and Hui Zhang. Csl: A large-scale chinese scientiﬁc literature dataset. arXiv preprint arXiv:2209.05034, 2022.

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

- [18] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281, 2023.

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

- [19] Dingkun Long, Qiong Gao, Kuan Zou, Guangwei Xu, Pengjun Xie, Ruijie Guo, Jian Xu, Guanjun Jiang, Luxi Xing, and Ping Yang. Multi-cpr: A multi domain chinese dataset for passage retrieval. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 3046–3056, 2022.

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

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

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

- [20] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

- [21] Julian McAuley and Jure Leskovec. Hidden factors and hidden topics: understanding rating dimensions with review text. In Proceedings of the 7th ACM conference on Recommender systems, pages 165–172, 2013.

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

- [22] Niklas Muennighoﬀ, Nouamane Tazi, Lo¨ıc Magne, and Nils Reimers. Mteb: Massive text embedding benchmark. arXiv preprint arXiv:2210.07316, 2022.

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

- [23] openai. text-embedding-v3. openai blogs, 2024. URL: https://openai.com/blog/new-embedding-models-and-api-updates.
- [24] Yifu Qiu, Hongyu Li, Yingqi Qu, Ying Chen, Qiaoqiao She, Jing Liu, Hua Wu, and Haifeng Wang. Dureader retrieval: A large-scale chinese benchmark for passage retrieval from web search engine. arXiv preprint arXiv:2203.10232, 2022.

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

- [25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

8763. PMLR, 2021.

- [26] Samyam Rajbhandari, Jeﬀ Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.

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

- [27] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, 2019.

[Figure 306]

[Figure 307]

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

- [28] Ye Liu Rui Meng. Sfr-embedding-mistral:enhance text retrieval with transfer learning. Salesforce AI Research Blog, 2024. URL: https://blog.salesforceairesearch.com/sfr-embedded-mistral.
- [29] Darius Koenig Sean Lee, Aamir Shakir. Open source strikes bread - new ﬂuﬀy embeddings model, 2024. URL: https://www.mixedbread.ai/blog/mxbai-embed-large-v1.

- [30] shibing624. Nlizh dataset, 2023. URL: https://huggingface.co/datasets/shibing624.
- [31] SophonPlus. Onlineshoppingdataset, 2016. URL: https://github.com/SophonPlus/ChineseNlpCorpus.
- [32] Jianlin Su. Cosent (1): A more eﬀective sentence vector scheme than sentence bert, Jan 2022. URL: https://kexue.fm/archives/8847.
- [33] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text embeddings by weaklysupervised contrastive pre-training. arXiv preprint arXiv:2212.03533, 2022.

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

- [34] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. Improving text embeddings with large language models. arXiv preprint arXiv:2401.00368, 2023.

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

- [35] Lucy Lu Wang, Kyle Lo, Yoganand Chandrasekhar, Russell Reas, Jiangjiang Yang, Douglas Burdick, Darrin Eide, Kathryn Funk, Yannis Katsis, Rodney Kinney, et al. Cord-19: The covid-19 open research dataset. ArXiv, 2020.

[Figure 358]

- [36] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighof. C-pack: Packaged resources to advance general chinese embedding. arXiv preprint arXiv:2309.07597, 2023.

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

- [37] Xiaohui Xie, Qian Dong, Bingning Wang, Feiyang Lv, Ting Yao, Weinan Gan, Zhijing Wu, Xiangsheng Li, Haitao Li, Yiqun Liu, et al. T2ranking: A large-scale chinese benchmark for passage ranking. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2681–2690, 2023.

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

- [38] Liang Xu, Hai Hu, Xuanwei Zhang, Lu Li, Chenjie Cao, Yudong Li, Yechen Xu, Kai Sun, Dian Yu, Cong Yu, et al. Clue: A chinese language understanding evaluation benchmark. arXiv preprint arXiv:2004.05986, 2020.

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

- [39] S. Zhang, X. Zhang, H. Wang, L. Guo, and S. Liu. Multi-scale attentive interaction networks for chinese medical question answer selection. IEEE Access, 6:74061–74071, 2018. doi:10.1109/ACCESS.2018.2883637.

[Figure 395]

[Figure 396]

- Table 7: Example of a Prompt to Generate a Topic Phrase in Phase 1, {Retrieval tasks[x1]} and {Retrieval tasks[x2]} are examples of randomized references, {NUM} indicates the number of generated topic phrases.

[Figure 397]

[Figure 398]

[Figure 399]

”Brainstorm a list of potentially useful text retrieval tasks. Here are a few examples for your reference:

- - {Retrieval tasks[x1]}

[Figure 400]

- - {Retrieval tasks[x2]} Please adhere to the following guidelines:

[Figure 401]

- - Specify what the text is, and what the desired documents are.
- - Each retrieval task should cover a wide range of queries, and should not be too speciﬁc. Your output must always be string, the string is a json dict start with { and ends with }, the key is ‘tasks’, and the value is a list of strings only, with about {NUM} elements, and each element corresponds to a distinct retrieval task in one sentence. Do not explain yourself or output anything else. Be creative!”

[Figure 402]

# Appendix

Prompt for the retrieval data

Referring to method [34], We adopted a two-stage approach to data generation. In the ﬁrst phase, we let GPT4 randomly generate a large number of topic phrases in various domains to ensure the diversity of the data. The prompt used is shown in the Table 7. In the second phase, we generate text pairs containing query, positive text and negative text based on the large number of topic phrases obtained from the previous generation through a specially designed prompt. In order to enrich the dataset, we generate text pairs 10 times for each topic phrase, and for each text pair, we set the corresponding random parameter in the prompt to make the text pairs obtained each time completely diﬀerent. For example, for query generation, we randomly set its word count, clarity, etc., and randomly set the length and comprehensibility of positive text and negative text. It should be mentioned that for the deﬁnition of negative text, we require that the result generated by GPT4 should contain some useful information in the query, but its usefulness and comprehensiveness should not exceed that of the positive text. The prompt used is shown in the Table 8.

- Table 8: Example of a prompt to generate text pairs containing query, positive text and negative text in Phase 2, “{task}” is the topic phrase generated in the ﬁrst phase, “{query type}” ∈ {extremely long-tail, long-tail, common}, “{query length}” ∈ {less than 5 words, 5 to 15 words, at least 10 words}, “{clarity}” ∈ {clear, understandable with some eﬀort, ambiguous}, “{num words}” ∈ {50, 100, 200, 300, 400, 500}, “{difficulty}” ∈ {high school, college, PhD}.

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

“You have been assigned a retrieval task: {task} Your mission is to write one text retrieval example for this task in JSON format. The JSON object must contain the following keys:

- - ‘user query’: a string, a random user search query speciﬁed by the retrieval task.

[Figure 407]

- - ‘positive document’: a string, a relevant document for the user query.

[Figure 408]

- - ‘hard negative document’: a string, a hard negative document that only appears relevant to the query. Please adhere to the following guidelines:

[Figure 409]

[Figure 410]

- - The ‘user query’ should be {query type}, {query length}, {clarity}, and diverse in topic.

[Figure 411]

[Figure 412]

[Figure 413]

- - All documents must be created independent of the query. Avoid copying the query verbatim. It’s acceptable if some parts of the ’positive-document’ are not topically related to the query.
- - All documents should be at least {num words} words long.

[Figure 414]

- - The ’hard negative document’ contains some useful information, but it should be less useful or comprehensive compared to the ’positive document’.

[Figure 415]

[Figure 416]

[Figure 417]

- - Both the query and documents should be in language.
- - Do not provide any explanation in any document on why it is relevant or not relevant to the query.
- - Both the query and documents require {diﬃculty} level education to understand. Your output must always be a JSON object only, do not explain yourself or output anything else. Be creative!”

[Figure 418]

