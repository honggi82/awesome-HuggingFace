Knesset-DictaBERT: A Hebrew Language Model for Parliamentary Proceedings

Gili Goldin University of Haifa gili.sommer@gmail.com

Shuly Wintner University of Haifa shuly@cs.haifa.ac.il

arXiv:2407.20581v1[cs.CL]30Jul2024

Abstract

We present Knesset-DictaBERT, a large Hebrew language model ﬁne-tuned on the Knesset Corpus, which comprises Israeli parliamentary proceedings. The model is based on the DictaBERT architecture and demonstrates signiﬁcant improvements in understanding parliamentary language according to the MLM task. We provide a detailed evaluation of the model’s performance, showing improvements in perplexity and accuracy over the baseline DictaBERT model.

# 1 Introduction

The ﬁeld of natural language processing (NLP) has seen remarkable advancements in recent years, driven by the development of large language models. These models have signiﬁcantly enhanced the ability to understand and generate human language. However, much of the effort of creating and training NLP models has been focused on English, while fewer NLP models are available for lower-resource languages such as Hebrew. We present one such model here.

Parliamentary proceedings are a valuable source of information for understanding political discourse, legislative processes, and more. Analyzing these texts requires models that can accurately capture the nuances of the language used in such settings. Despite the importance of this task, there has been a lack of specialized models trained on parliamentary corpora in Hebrew. To address this gap, we ﬁne-tuned the pre-trained Dicta-BERT model (Shmidman et al., 2023) on the Knesset Corpus (Goldin et al., 2024), a dataset of Israeli parliamentary proceedings. DictaBERT is a state-of-the-art Hebrew language model based on the BERT architecture (Devlin et al., 2019), which was trained on a diverse set of Hebrew texts. Knesset-DictaBERT is the resulting ﬁnetuned model: it is speciﬁcally tailored for Hebrew

parliamentary text. In this paper we describe the training process of this model, provide a detailed evaluation of its performance, and demonstrate its superiority over the baseline DictaBERT model on the Knesset data. We believe that KnessetDictaBERT will be a valuable resource for researchers and other users working on Hebrew language processing and political text analysis.

# 2 Methodology

We ﬁne-tuned the pre-trained DictaBERT model, a state of the art language model for Hebrew, specifically for the masked language modeling (MLM) task, on the full Knesset Corpus dataset (Goldin et al., 2024). The model was initialized using the pre-trained weights from the dicta-il/dictabert checkpoint available on Hugging Face. 2.1 Dataset and Data preprocessing

We used the Knesset Corpus, which is is a large Hebrew dataset of Israeli parliamentary proceedings, as the dataset for ﬁne-tuning DictaBERT. The corpus contains over 32M sentences, over 384M tokens and comprises texts from both plenary and committee protocols.

The corpus was preprocessed to create text shards for efﬁcient loading and processing. We split the dataset into training, validation, and test sets with ratios of 80%, 10%, and 10%, respectively.

First, we tokenized the input text using the AutoTokenizer from Hugging Face’s transformers library. Each text sample was tokenized into sequences of tokens, with each token represented by its corresponding ID from the tokenizer’s vocabulary. To prepare the data for the MLM training, we grouped the tokenized texts into ﬁxed length chunks of 256 tokens. The last chunk was padded to ensure consistent input sizes for the model. We used the DataCollatorForLanguageModeling from

the transformers library to dynamically mask tokens during training, with a masking probability of 15%.

2.2 Training procedure and hyperparameters We trained the model on a SLURM (Simple Linux Utility for Resource Management) environment. We utilized a distributed training setup with the NCCL backend to leverage multiple GPUs, ensuring efﬁcient training and gradient synchronization.

We used the following training conﬁguration: We used a per-device batch size of 32. In order to effectively double the batch size without increasing memory usage, we set the gradient accumulation steps to 4. A learning rate of 1e−4 was chosen for the AdamW optimizer and a weight decay of 0.01 was applied to regularize the model. The model was trained for 2 full epochs. We enabled ‘fp16’ (mixed precision) to speed up training and reduce memory usage. Mixed precision training uses 16-bit ﬂoating-point numbers instead of the standard 32-bit, which can signiﬁcantly improve computational efﬁciency and decrease memory usage (Micikevicius et al., 2017). Periodic evaluations were conducted on the validation set, and the best-performing model checkpoint was identiﬁed based on the validation loss. Final evaluations were performed on the test set to assess the model’s performance. The model is available on Hugging Face hub at Knesset-DictaBERT.

# 3 Experiments and Results

The ﬁne-tuned Knesset-DictaBERT model was evaluated on the test set, which contained about 3.2 million sentences (about 38 million tokens), using perplexity as the primary metric. The model achieved a perplexity of 6.60, signiﬁcantly outperforming the original DictaBERT model, which showed a perplexity of 22.87. Evidently, the Knesset-DictaBERT model reﬂects the language of Knesset proceedings much better than the baseline DictaBERT model.

We also evaluated the ﬁne-tuned KnessetDictaBERT model for its accuracy in predicting masked tokens on a subset of the test set containing approximately 300K sentences (about 3.5 million tokens). Since MLM is considered a challenging task, where multiple options may be equally plausible, the evaluation focused on three accuracy metrics: top-1 accuracy, top-2 accuracy, and top5 accuracy. These metrics measure the model’s

ability to correctly predict the masked token as the ﬁrst, within the ﬁrst two, or within the ﬁrst ﬁve predictions, respectively.

The Knesset-DictaBERT model correctly identiﬁed the masked token in the top-1 prediction 52.55% of the time, compared to the original Dicta model, which achieved a top-1 accuracy of 48.02%. Additionally, Knesset-DictaBERT succeeded in cases where the original DictaBERT model did not a total of 52,464 times. In contrast, the original DictaBERT model succeeded where Knesset-DictaBERT did not in only 27,995 times. Furthermore, when considering the top-2 predictions, Knesset-DictaBERT correctly identiﬁed the masked token 63.07% of the time, whereas the original DictaBERT model had a top-2 accuracy of 58.60%. Moreover, Knesset-DictaBERT succeeded in 19,400 instances where the original model failed to provide a correct prediction within the top-2, while the original DictaBERT model, succeeded in only 13,862 instances where KnessetDictaBERT did not. On top of that, when extending the scope to the top-5 predictions, KnessetDictaBERT demonstrated a signiﬁcant improvement with a 73.59% accuracy, while the original DictaBERT model achieved a 68.98% accuracy.

In all tested metrics the Knesset-DictaBERT model outperformed the original DictaBERT model, indicating a more robust performance in predicting masked tokens within parliamentary text. These results highlight the effectiveness of ﬁne-tuning on the speciﬁc parliamentary corpus. The results are presented in Table 1.

# 4 Conclusion and Future Work

In this work, we successfully ﬁne-tuned the DictaBERT model on the Knesset Corpus to create Knesset-DictaBERT, a model proﬁcient at understanding and generating parliamentary language in Hebrew. The results indicate a robust model performance, with substantial improvements over the baseline model. Future work may involve evaluation on additional Hebrew datasets to enhance the model’s generalization capabilities and ﬁne-tuning other language models on the Knesset corpus.

# Limitations

The model was ﬁne-tuned speciﬁcally on the Knesset Corpus, which comprises parliamentary proceedings. As a result, its performance on general Hebrew text or other domains may not be as

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Metric Knesset-DictaBERT Original DictaBERT Perplexity 6.60 22.87

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

- Top-1 Accuracy 52.55% 48.02%

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

- Top-2 Accuracy 63.07% 58.60% Top-5 Accuracy 73.59% 68.98%

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Table 1: Comparison of Knesset-DictaBERT and Original DictaBERT on perplexity and accuracy metrics

robust. However, the original DictaBERT model was trained on a variety of resources in Hebrew, which probably allows the Knesset-DictaBERT to still beneﬁt from the diverse linguistic patterns and vocabulary present in the broader training data of the original model.

# Ethical Considerations

The Knesset Corpus may contain inherent biases, reﬂecting the political and social biases present in parliamentary discussions. Consequently, Knesset-DictaBERT may inherit these biases.

# References

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Gili Goldin, Nick Howell, Noam Ordan, Ella Rabinovich, and Shuly Wintner. 2024. The Knesset Corpus: An annotated corpus of Hebrew parliamentary proceedings. ArXiv, abs/2405.18115.

Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory F. Diamos, Erich Elsen, David García, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. 2017. Mixed precision training. CoRR, abs/1710.03740.

Shaltiel Shmidman, Avi Shmidman, and Moshe Koppel. 2023. DictaBERT: A state-of-the-art BERT suite for Modern Hebrew.

This figure "committee_ttr.png" is available in "png"  format from:

http://arxiv.org/ps/2407.20581v1

