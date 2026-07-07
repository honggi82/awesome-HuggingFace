# arXiv:2310.09263v1[cs.CL]13Oct2023

## Table-GPT: Table-tuned GPT for Diverse Table Tasks

Peng Li†, Yeye He‡, Dror Yashar, Weiwei Cui, Song Ge, Haidong Zhang, Danielle Rifinski Fainman, Dongmei Zhang, Surajit Chaudhuri

Microsoft Corporation

### ABSTRACT

Language models, such as GPT-3 and ChatGPT, demonstrate remarkable abilities to follow diverse human instructions and perform a wide range of tasks. However, when probing language models using a range of basic table-understanding tasks, we observe that today’s language models are still sub-optimal in many table-related tasks, likely because they are pre-trained predominantly on onedimensional natural-language texts, whereas relational tables are two-dimensional objects.

In this work, we propose a new “table-tuning” paradigm, where we continue to train/fine-tune language models like GPT-3.5 and ChatGPT, using diverse table-tasks synthesized from real tables as training data, with the goal of enhancing language models’ ability to understand tables and perform table tasks. We show that our resulting Table-GPT models demonstrate (1) better table-understanding capabilities, by consistently outperforming the vanilla GPT-3.5 and ChatGPT, on a wide-range of table tasks, including holdout unseen tasks, and (2) strong generalizability, in its ability to respond to diverse human instructions to perform new table-tasks, in a manner similar to GPT-3.5 and ChatGPT.

### 1 INTRODUCTION

Large language models, such as GPT and LLaMa, have recently demonstrated impressive abilities in performing diverse naturallanguage tasks [5, 8, 12, 50]. In the database literature, a number of pioneering work, such as [20, 29, 39, 43], have also shown that by using “prompt engineering”, to careful select the best instructions and few-shot examples for a particular task at hand, language models can be prompted to perform well on a number of table-tasks such as entity matching and data-imputation.

While prompt-engineering is a promising direction to enhance modelperformance,itrequirestask-specific tuning (e.g., task-specific labeled-data to test the performance of different instruction/example combinations) [6, 8, 61]. We in this work propose an orthogonal paradigm called “table-tuning”, where instead of modifying prompts, we modify the weights of the underlying language models for once (i.e., not task-specific), by continuing to train them using

†: Affiliation: Georgia Tech (pengli@gatech.edu), work done at Microsoft. ‡: Correspondence: yeyehe@microsoft.com

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Conference’17, July 2017, Washington, DC, USA © 2023 Association for Computing Machinery. ACM ISBN 978-x-xxxx-xxxx-x/YY/MM...$15.00 https://doi.org/10.1145/nnnnnnn.nnnnnnn

[Figure 1]

Figure 1: Two simple tests to probe language-models’ basic ability to read and understand tables. (Left) T-1: Missing cells identification, which is to identify the column-header/rowid of a missing cell. (Right) T-2: Column-Finding, which is to identify the column-name of a given value. Even large models (e.g. 175B GPT-3.5) can frequently fail on such tests, with only 0.26 accuracy in one variant of the tests.

diverse table-tasks as training data, to improve their ability to understand tables. We show that table-tuned Table-GPT consistently outperform the vanilla GPT-3.5 and ChatGPT on a wide-range of table tasks, including new and unseen table-tasks. We note that our model-tuning approach is complementary to prompt-engineering, because carefully engineered prompts can continue to benefit both vanilla language-models and our table-tuned models.

Today’s language models cannot “read tables” reliably. While today’s language models excel in natural-language tasks, we start by asking the question of whether these models are optimal for table-tasks, because after all, they are pre-trained predominantly on natural language texts, which are different from tables.

More specifically, natural language texts are (1) one-directional,

- (2) read left-to-right, where (3) swapping two tokens will generally change the meaning of a sentence. In contrast, relational tables are

(1)two-dimensional in nature with both rows andcolumns, (2) where reading top-to-bottom in the vertical direction for values in the same column, is crucial in many table-tasks. Furthermore, unlike text,

- (3) tables are largely “invariant” to row and column permutations, where swapping two rows or columns do not generally change the semantic meaning of the table.

With this question in mind, we perform two simple tests to probe language models’ ability to “read” tables and then answer basic questions, which we call (T-1) Missing-value-identification, and (T-2) Column-finding, as shown in Figure 1.

In (T-1) Missing-value-identification, we show language models with a real table, presented in a markdown or alternative format, where we make sure that there is exactly one empty cell in the table.

Markdown table is a common format used by prior work to feed tables into languagemodels, and also a format that models like GPT will use when it needs to respond with a table, presumably because GPT-like models use GitHub data in its pre-training, where markdown-format tables are abundant.

[Figure 2]

change the semantic meaning of the table, at least to humans. This is presumably because language-models are pre-trained on text where the order of tokens matters (e.g., “Jennifer called you” vs. “you called Jennifer”), leading to sub-optimal behaviors on tables.

We believe observations like these point to opportunities for us to improve the underlying language model, by enhancing their ability to understand tables and perform table-tasks.

Instruction-tuning in NLP: train language-models to follow diverse human instructions. To change the behaviour of language models, successful attempts have been made in the NLP community, using a technique known as “instruction-tuning” in the literature [40, 47, 57–59].

- Figure 2: Example table-tasks, where the ability of language models to “read” tables vertically is important. (Left) T-3: Table Question-Answering. (Right) T-8: Data Imputation. More tasks like these are shown in Table 2.

It was observed in the NLP community [8, 40, 59], that earlier versions of pre-trained language models, such as GPT-3, is able to complete a sentence with the next likely token (e.g., “write a bed-time” → “story”), but cannot reliable follow higher-level instructions from humans (e.g., “write a bed-time story for a 3 years-old, in 100 words”), a behavior that is only demonstrated in later models such as ChatGPT.

We then ask the model to identify the empty cell, by responding with the column-name and row-id of the empty cell, repeating for 1000 randomly sampled real tables. Despite the impressive ability of language-models like GPT-3.5 to perform diverse tasks, we find that they fail on a surprisingly large fraction (up to 74%) of such tests, often responding with incorrect column-headers or row-ids – for instance, in the example shown in Figure 1, the model may answer that the column “music” has a missing cell, when the correct answer should be “art”.

Instruction-tuning was the key technique invented that continues to train GPT-like models into ChatGPT-like models, in a process shown on the left of Figure 3. Diverse training data in the form of “(instruction, completion)” pairs are constructed, often manually annotated by human labellers [40], e.g. (“write a bed-time story” → an-actual-story), to continue train language-models on these explicit demonstrations of how to follow high-level human instructions, leading to well-known models such as ChatGPT/InstructGPT [2, 40], as well as their open-source counterparts like Stanford-Alpaca [4] and LLaMa-chat [50].

In order to ensure that there is no ambiguity in what “missing value” or “empty cell” could mean to language models, we design a second and even simpler test, which we refer to as: (T-2) Columnfinding, shown on the right of Figure 1. In this test, we present a language model with a real table, and ask it to find a specific cell-value that appears exactly once in the entire table (e.g., “93” in this example), and then respond with the column-name of the that value. We find that language models such as GPT-3.5 are prone to fail on such tests again (e.g., answering that “93” is in column “art” when the correct answer is “music”), on over half of such tests.

Table-tuning: train language-models to understand tables. We believe that the research on instruction-tuning in NLP, which successfully enhances language-models ability to follow human instructions, holds lessons for us when we aim to enhance languagemodels ability to understand tables and perform table-tasks.

We believe these simple probes show that today’s large language models, when pre-trained on large amounts of one-directional natural-languagetexts,arenotbest-suited to “read” two-dimensional tables, especially in the vertical direction, which however is crucial in performing many table-tasks.

In this work, we propose a “table-tuning” paradigm analogous to instruction-tuning, where we continue to train language-models, using diverse training data in the form of (instruction, table, completion), which we synthesize using large amounts of real tables. This process is illustrated on the right of Figure 3.

Consider, for example, the popular NLP task of (T-3) TableQA [11, 42, 49], where the task is to answer a natural-language question, based on the content of the table. The left side of Figure 2 shows such an example. Given the question “How many second-graders scored over 90 in art, in the table below?” Imagine that a model is not able to “read” tables correctly, it may reason that both “Jennifer” and “James” satisfy the condition (because it believes “93” is in the column “art”, like shown in Figure 1 (Right)), and may answer “2” instead of the correct “1”. We emphasize that the ability to read in the vertical direction (top-to-bottom for values in the same column) is similarly important in many other tabletasks, such as data-imputation (shown on the right of Figure 2), data-transformation, error-detection, NL-to-SQL, etc., like the list in Table 2 would show, which includes a diverse set of table-tasks considered in this work.

Through extensive experiments, we show that “table-tuning” is a promising new direction, as our resulting Table-GPT models are:

- (1)Strong table models,whichsubstantiallyoutperform 175B GPT3.5 and ChatGPT, on a wide range of seen and unseen table-tasks, as we summarize in Table 2 and Figure 9;

- (2) Generalizable to new tasks, as they can respond well to novel and unseen table-tasks, similar to how Chat-GPT could generalize and respond to new and unseen NLP tasks, like shown in Figure 4.

Contributions. We make the following contributions:

- • We propose a new “table-tuning” paradigm to continue to train language models, specifically designed to enhance languagemodels’ ability to perform table-tasks, using diverse table-tasks synthesized from large amounts of real tables, in a “synthesisthen-augment” process.
- • We developtask-level,table-level,instruction-level, and completionlevel data augmentation techniques for table-tuning, which we

In addition, we find that large language models are sensitive to the order in which columns are presented in a table – e.g., when we swap the order of two columns in a table, a model can change its response for a table-task, even when such a swap should not

[Figure 3]

- Figure 3: Instruction-tuning vs. Table-tuning. (Left) Instruction-tuning is a technique developed in the NLP community that continues to train language-models (e.g., GPT) for instruction-following capabilities (e.g., in ChatGPT). (Right) Table-tuning is an analogous approach we propose to train language-models to better understand table and perform table-tasks.

show are crucial to avoid over-fitting and ensure the generality of Table-GPT.

• We show that Table-GPT not only excels on table-tasks in both zero-shot and few-shot settings out of box, but can also serve as a “table foundation model” and used as a better starting point than vanilla GPT, for down-stream single-task optimizations such as task-specific fine-tuning and prompt-engineering.

### 2 PRELIMINARIES

We will start with a review of language models, and then the use of language models in table-tasks.

### 2.1 Language models

There are two popular styles of language models today, known as the decoder and encoder-style, both derived from the original transformer architecture [53].

Encoder-style language models. One class of popular languagemodels,including thewell-known BERT [17]andRoBERTa[35], use only encoders from the transformer, and are pre-trained on large amounts of texts to effectively represent the semantics of texts using embedding vectors.

Down-stream tasks: Task-specific fine-tuning. To use encoderstyle models like BERT for downstream tasks, task-specific finetuning is generally employed [21, 34], which continues to fine-tune (or train) BERT-like models for a given task, using task-specific labeled data. For example, suppose the downstream task is sentiment analysis of Yelp restaurant reviews, then labels in the form of (“The food is amazing”, “positive”), (“The service is slow”, “negative”), are needed to fine-tune BERT-like models for the desired outcome [17, 46].

Crucially, when the target input data or the desired output changes, the labeling effort often needs to repeat for the best performance. For example, if the input data for sentiment analysis changes to IMDB reviews, or if the output needs to include a classification of “cuisine-type” for restaurant reviews. While encoderstyle language-models are strong models, the need to fine-tune with task-specific labelled data limits its ability to generalize to new unseen tasks [17, 22, 35, 46].

Decoder-style “generative” language models. Another class of decoder-only language models, such as GPT [8] and LLaMa [50], are generative in nature, and are shown to excel in generalizing to new downstream tasks without task-specific fine-tuning [8].

Generalize to new tasks: zero-shot and few-shot learning.It was shown in the NLP literature that the decoder-style models (e.g., GPT and LLaMa), especially after instruction-tuning [31, 40, 47, 56– 59, 67] (e.g., ChatGPT/InstructGPT [2, 40] and Stanford Alpaca [4]), can adapt to new tasks easily, using just natural-language instructions (e.g., “classify the sentiments in the following reviews”), and optionally a few examples. Such an approach can adapt to new datasets (e.g., IMDB vs. Yelp reviews) and new tasks (sentimentanalysis vs. machine-translation), without fine-tuning on labelled data for each specific task, making the decoder-style models more general and versatile. Figure 5 shows the benefit of “instructiontuning” in model generalizability, depicted pictorially on the y-axis.

### 2.2 Language models for table tasks

Pioneering work in the database literature have employed language models in various ways to perform table-related tasks.

Encoder-style language models for table tasks. There is a long and fruitful line of research (e.g., TURL [16], TaBERT [64], Ditto [32] and Doduo [48]), where table-models are trained based

[Figure 4]

[Figure 5]

- Figure 4: Table-models should ideally “generalize” to new datasets and new tasks. (Left) Column type annotation (CTA): while this is a common table-task, the list of target-types to choose from can vary from dataset to dataset (e.g., 78 types in [25], and 107 in [16]). Making table-models to “generalize” to new CTA dataset without needing to retrain, is useful. (Right) Text-to-Table: a general table-model should be

Figure 5: Instruction-tuning vs. Table-tuning. Instructiontuning improves model “generalizability”, to follow diverse human-instructions to perform new and unseen tasks (xaxis), whereas our proposed table-tuning is analogous in spirit but aims to improve model ability to understand tables and perform table-tasks (y-axis).

- as general-purpose as models like ChatGPT, in following instructions to perform novel unseen table-tasks, such as “extracting tables from text” in the example.

fine-tuning), just like the underlying GPT-3 and ChatGPT models. In other words, in Table-GPT we aim to get the “best of both worlds”, with both generalizability, and good table-task performance.

on encoder-style BERT-like models, which are shown to perform well on various table tasks.

However, like their BERT-like base models, to generalize to a new dataset or a new task, these encoder-style table-models generally require fine-tuning with labeled data. As a concrete example, for the table-task of “column-type-annotation” [16, 48], in order to move from one dataset with 78 semantic types [25], to another dataset with 107 semantic types [16], new labeled data have to be obtained, so that the models can be fine-tuned to generate the new output with 107 classes [16]. In contrast, being able to adapt to new datasets and tasks without task-specific fine-tuning, is a key goal that we want to achieve in this work, like illustrated in Figure 4.

### 3 CAN LANGUAGE MODELS “READ” TABLES?

Since language models like GPT are pre-trained predominantly on natural language text, we start by asking a basic question of whether language models can reliable read and understand relational tables, which are different from text in many ways, as we discuss below.

One-dimensional (text) vs. two-dimensional (tables). Language models trained mostly on natural language text (e.g, books and web pages) and programming code (e.g., GitHub), both of which that are one-directional that is meant to be read left-to-right, toke-by-token, in a sequential manner.

Decoder-style language models for table tasks. With the success of decoder-style language models such as GPT-3 and ChatGPT, which are shown to perform tasks out-of-the-box with instructions only, pioneering research in the database field develop “promptengineering” techniques for table-tasks [29, 39, 43], which carefully selects instructions and examples in the prompt, such that vanilla language models can perform well on table-related tasks.

In contrast, relational tables are two-dimensional with rows and columns, where reading top-to-bottom vertically, for columnheaders and other values in the same column (which may be far away when a table is serialized), is crucial for many table-tasks.

Consider the task of Data-Imputation [7, 37] (T-8 in Table 2), which is to infer a missing value in a table cell, like shown in the example of Figure 2 (Right). At least for humans, it is natural to look vertically in the horizontal direction, to see the column-header (“continent” in this case), as well as other values in the same column (e.g., “Americas”), before one can make a guess for the missing value.

Table-tuning for table-tasks. In contrast to prompt-engineering that optimizes prompts, our proposed “table-tuning” explores the orthogonal direction, where we continue to train the underlying language models, for once only (not task-specific), so that the resulting model perform better on a range of table-tasks. This is complementary to prompt-engineering, because carefully-engineered instructions and examples can continue to benefit both the vanilla GPT as well as our Table-GPT, as we will show in our experiments.

Similarly, for the task of Error-Detection [14] (T-9 in Table 2) it is also necessary to look at the column-header and other values in the same column, to understand the semantics of the column, before one can determine if a cell is erroneous.

- Figure 5 shows the process of table-tuning, which is analogous

Even for table-tasks that may be a bit removed, such as Table Question-Answering [42, 49] (T-3 in Table 2), which is traditionally an NLP problem – examples like in Figure 2 (Left) would show that, in order to answer a question correctly on a table, reading vertically in a column (e.g., for values in the art) is similarly important.

to instruction-tuning, but unlike instruction-tuning that improves model generalizability to follow human instructions (y-axis), we focus on improving underlying models ability to understand tables and perform table-tasks (x-axis). Crucially, as we will show, our table-tuned models remain to be general and capable of following human-instructions to perform table-tasks (without task-specific

To test language models’ ability to read tables in the columnar direction, we design simple tests. In the first test, referred to as

[Figure 6]

- Figure 6: Two variants of the task (T-1) Missing-cellidentification. (Left) T-1(a): We remove a random cell from a real table, but keep its column-separator. The presence of “| |” indicates a missing cell, which should be easy to identify. (Right) T-1(b): We remove a random cell, as well as its column-separator, which is a common but challenging CSV parsing issue [18, 52, 54].

|(T-1): Missing cell|Find col-header tests: (a) no col-sep (b) has col-sep<br><br>| |Find row-id tests: (a) no col-sep (b) has col-sep| |
|---|---|---|---|---|
| | | | | |
|GPT-3.5 (zero-shot)<br><br>|0.26|0.30<br><br>|0.76|0.87|
|GPT-3.5 (few-shot)<br><br>|0.38<br><br>|0.51<br><br>|0.77<br><br>|0.91|

- Table 1: Accuracy numbers of vanilla GPT-3.5 (using Text-Davinci-002), on the task of (T-1) Missing-valueidentification as shown in 6.

“Missing-value-identification” (T-1 in Table 2), we sample a real table𝑇 with no missing cells, and remove a random cell from𝑇. We then produce two variants of the test, like shown in Figure 6:

- T-1(a): we keep the column separator of the missing cell and ask language-models to identify the row-id/column-header of the missing cell, like in Figure 6 (Left), which seems simple;

- T-1(b): We remove the column separator of the missing cell and then ask the same question, like in Figure 6 (Right). This is a common situation in CSV parsing that can be challenging [18, 52, 54], as one needs to align values vertically to see the missing value is in which column. (In the case, humans can see that the countries “USA” and “China” should align, the GPD numbers should align, so there must be a missing cell in “row-2”, in between “China” and “19,373,586”, for the column “Continent”).

We repeat these two tests 1000 times, using 1000 randomly sampled real tables. Table 1 shows the result of this test. We can see that it is clearly challenging for language models to read tables in the column direction, where the accuracy with and without columnseparator is 0.38 and 0.26, respectively. Even with column-separator and explicit few-shot demonstrations, the model is only able to get half of the tests right (0.51).

In the row-direction, the model’s ability to identify a missing cell is clearly better, though still not great, especially in the “no col-separator” setting.

To ensure that the language models are not confused by what we mean in “missing cell”, we create a second, even simpler test, called Column-Finding (T-2 in Table 2), shown with an example in Figure 1 (Right), where we ask the model to find the column-header of a specific value, which appears exactly once in a given table 𝑇, for 1000 randomly sampled real tables. Our result show that the accuracy of GPT-3 is similarly low (0.46), confirming the hypothesis that language models ability to read two dimensional tables is likely insufficient.

Order-sensitive (text) vs. permutation-invariant (tables). In addition, we observe that natural-language texts tend to be ordersensitive, where swapping two tokens will generally lead to different meanings (e.g., “Jennifer called you” vs. “you called Jennifer”). In comparison, tables tend to be permutation-invariant, where swapping two rows or two columns, should generally not change the semantic meaning of the resulting table.

As a result, when applying language-models to table-tasks, e.g., Entity-matching, Error-Detection, Schema-Matching, we find that the predictions can be rather sensitive to the order in which columns are presented in the input tables, even when we only slightly reorder the columns.

We believe it shows that language models understanding of tables is still unstable and sub-optimal (likely influenced by the overwhelming text data used in its pre-training), because the decisions for tasks like Entity-matching and Error-Detection should really not depend on the order of columns.

Other differences. There are a number of additional aspects that make tables different from text. For example, table-cells tend to be short-form entity-names or phrases, which when serialized in a row, will typically be different from natural-language sentences found in text documents. Furthermore, values in the same column generally have homogeneous values, with pairs of columns encode regular relationships, which is another property not found in texts. All of these make tables different from texts, likely rendering languagemodels sub-optimal for table use cases, which motivates our tabletuning approach described next.

### 4 TABLE-TUNING FOR TABLE-GPT

We propose a new table-tuning paradigm, to enhance language models ability to understand tables and perform table-tasks,

### 4.1 Overall approach: Synthesis-then-Augment

Like discussed earlier, our table-tuning is inspired by the success of “instruction-tuning” from the NLP literature [40, 57, 59], illustrated in Figure 3 (Left), where diverse training data in the form of “(instruction, completion)” pairs are used to continue to train language-models, and has led to popular models like ChatGPT and LLaMa-chat that can understand and follow human instructions.

Our proposed table-tuning, as illustrated in Figure 3 (Right), is similar in spirit – instead of improving language-model ability to follow instructions using diverse “(instruction, completion)” pairs, we aim to improve language-model ability to perform table tasks using diverse “(instruction, table, completion)” triples, where each such triple defines an instance of a table-task:

Definition 1. An instance of a table-task, denoted by𝑡, is defined as a triple 𝑡 = (𝐼𝑛𝑠,𝑇,𝐶), where 𝐼𝑛𝑠 is the natural-language instruction that specifies the table-task,𝑇 is the input table on which the task is to be performed, and 𝐶 is the expected completion from following the instruction 𝐼𝑛𝑠 and performing the task on table𝑇.

Example 1. The examples in Figure 1, Figure 2, and Figure 3, show simple examples of table-tasks, defined by the (𝐼𝑛𝑠,𝑇,𝐶) triples, which correspond to (instruction, table, completion), respectively. Note that the completion 𝐶 can be natural-language texts (with JSON or other alternatives for answer parsing), tables, or a combination of both.

|Task-name|Task description (related work)<br><br>|Task category|Table data<br><br>|Train/Test|
|---|---|---|---|---|
|T-1: Missing-value identification (MV)<br><br>|Identify the row and column position of the only missing cell in a given table|Table understanding|synthesized<br><br>|Test only|
|T-2: Column-finding (CF)|Identify the column-name of a specific value that appears only once in a given table|Table Understanding|synthesized<br><br>|Test only|
|T-3: Table-QA (TQA)<br><br>|Answer a natural-language question based on the content of a table ([11, 42, 49])<br><br>|Table QA|[42]<br><br>|Test only|
|T-4: Column type annotation (CTA)<br><br>|Find the semantic type of a column, from a given list of choices ([16, 25, 63])|Table understanding<br><br>|[16, 25]|Test only|

|T-5: Row-to-row transform (R2R)<br><br>|Transform table data based on input/output examples ([23, 24, 27])|Data transformation<br><br>|synthesized (test: [24])|Train/Test|
|---|---|---|---|---|
|T-6: Entity matching (EM)|Match rows from two tables that refer to the same real-world entity ([32, 38, 41, 66])<br><br>|Table matching<br><br>|[1]|Train/Test|
|T-7: Schema matching (SM)<br><br>|Match columns from two tables that refer to the same meaning ([30, 36, 44])<br><br>|Table matching|synthesized (test: [30])<br><br>|Train/Test|
|T-8: Data imputation (DI)<br><br>|Predict the missing values in a cell based on the table context ([7, 37])|Data cleaning<br><br>|synthesized|Train/Test|
|T-9: Error detection (ED)|Detect data values in a table that is a likely error from misspelling ([14, 45])<br><br>|Data cleaning|synthesized<br><br>|Train/Test|

|T-10: List extraction (LE)<br><br>|Extract a structured table, from a list that lacks explicit column delimiters [9, 13, 19]|Data transformation<br><br>|synthesized|Train only|
|---|---|---|---|---|
|T-11: Head value matching (HVM)<br><br>|Match column-headers with its data values drawn from the same table|Table matching|synthesized<br><br>|Train only|
|T-12: Natural-language to SQL (NS)<br><br>|Translate a natural-language question on a table into a SQL query ([62, 65])<br><br>|NL-to-SQL|[65]|Train only|
|T-13: Table summarization (TS)|Produce a natural-language summary for the content in a table|Data augmentation<br><br>|synthesized<br><br>|Train only|
|T-14: Column augmentation (CA)<br><br>|Augment a table with additional columns compatible with a given table<br><br>|Data augmentation|synthesized<br><br>|Train only|
|T-15: Row augmentation (RA)|Augment a table with additional rows compatible with a given table<br><br>|Data augmentation|synthesized<br><br>|Train only|
|T-16: Row/column swapping (RCSW)|Manipulate a given table, by swapping the position of two rows or columns|Table manipulation<br><br>|synthesized|Train only|
|T-17: Row/column filtering (RCF)<br><br>|Manipulate a given table, by filtering on given rows or columns|Table manipulation<br><br>|synthesized|Train only|
|T-18: Row/column sorting (RCS)|Manipulate a given table, by performing sorting on given rows or columns<br><br>|Table manipulation<br><br>|synthesized<br><br>|Train only|

- Table 2: A summary of 18 table-related tasks, which we collect and synthesize, in order to “table-tune” GPT into Table-GPT. [Task categories]: These tasks cover diverse areas such as: table understanding, table-QA, table matching, table cleaning, table transformation, etc. Some of these tasks (T-1 to T-4) are used as unseen hold-out tasks, to evaluate Table-GPT ability to generalize to completely new and unseen tasks. [Table Data]: we choose to “synthesize” table tasks from diverse real tables when possible (e.g., when ground-truth can be produced automatically), to ensure the diversity of the training data and avoids over-fitting. When the ground-truth cannot be automatically produced (e.g., entity-matching, table-QA, NL-to-SQL, etc.), we use existing benchmark data from the literature.

The challenge, however, is that prior work on instruction-tuning have shown that the quality of the “(instruction, completion)” pairs is crucial [40, 50], to the extent that companies hired armies of human labelers to manually label such data, (e.g., instruction: “write a bed-time story with a bear goes to beach”, completion: an-actual-story-with-bears) [40], to ensure the quality and diverse of the training data.

We would like to replicate the success of instruction-tuning in the table domain, but ideally without the expensive human labeling.

Reusing existing benchmark data: insufficient diversity.One approach to generate table-tasks, is to use existing benchmark data

published in the database literature (similar efforts were made in the NLP literature for instruction-tuning [59]).

However, we found that the existing benchmark data to have:

- (1) limited task-diversity: as the literature tends to focus on a few select table-tasks that are hard and challenging (e.g., entitymatching and data-transformation); and
- (2) limited data-diversity: as benchmark data are typically labeled manually by researchers, only on a few specific datasets, which is sufficient for benchmark evaluation purposes, but insufficient when we want to use them as “training data” for language models.

Our attempt to use only existing benchmark data for table-tuning leads to over-fitting, due to the lack of task and data diversity.

Algorithm 1: Synthesize table-tasks for table-tuning

input :A corpus of diverse real tables C, a set of table-task types S output :Diverse synthesized table-tasks 𝐴 = {(𝐼𝑛𝑠,𝑇,𝐶)}

- 1 𝐷 ← {},𝐴 ← {}
- 2 foreach𝑇 ∈ C,𝑆 ∈ S do

- 3 (𝐼𝑛𝑠,𝑇,𝐶) ← Synthesize-Table-Task(𝑆,𝑇 ) // (Section 4.2)
- 4 𝐷 ← 𝐷 ∪ (𝐼𝑛𝑠,𝑇,𝐶)
- 5 foreach (𝐼𝑛𝑠,𝑇,𝐶) ∈ 𝐷 do

- 6 𝐼𝑛𝑠‘ ← Augment-Instruction(𝐼𝑛𝑠) // (Section 4.3)
- 7 𝑇‘ ← Augment-Table(𝑇 ) // (Section 4.3)
- 8 𝐶‘ ← Augment-Completion(𝐶) // (Section 4.3)
- 9 𝐴 ← 𝐴 ∪ (𝐼𝑛𝑠′,𝑇′,𝐶′)
- 10 return 𝐴

Our approach: Synthesis-then-Augment. We therefore propose a “synthesize-then-augment” approach to create diverse table-tasks using real tables, which can be used as training-data to table-tune language-models.

We show the main steps of our synthesize-then-augment approach in Algorithm 1. First, we sample a table 𝑇 ∈ C from a large corpus of real tables C, and a type of table-task 𝑆 ∈ S. From the (𝑇,𝑆) pair, we synthesize an instance of a table-task 𝑡 = (𝐼𝑛𝑠,𝑇,𝐶) (line 3), which is the task-synthesis step we will discuss in detail in Section 4.2. From the set of diverse instances of table-tasks created (𝐼𝑛𝑠,𝑇,𝐶), we then proceed to “augment” the tasks, at instruction/table/completion levels (line 6-8), which is a step that we will describe in Section 4.3. The resulting table-tasks𝐴 = {(𝐼𝑛𝑠′,𝑇′,𝐶′)} become the training data we use to table-tune language-models.

### 4.2 Synthesize diverse table-tasks

We now describe how we synthesize diverse instances of table-tasks 𝑡 = (𝐼𝑛𝑠,𝑇,𝐶) (Line 3 of Algorithm 1), so as to exercise languagemodels ability to understand two-dimensional table structures.

We propose two complementary approaches that (1) synthesize new table-tasks for task-diversity, and (2) synthesize new table test-cases of existing table-tasks for data-diversity. We will discuss each below in turn.

Synthesize new table-tasks for task-diversity. Since our goal is to enhance language models’ ability to understand tables, we believe it is not necessary to focus exclusively on challenging tabletasks that have been the focus of the literature [45]. Instead, we propose a number of table-understanding/augmentation/manipulation tasks that are easy to synthesize, leveraging large amounts of real tables that already exist. Specifically, we crawled 2.9M high-quality web-tables (e.g., Wikipedia) [10], referred to as C𝑤𝑡, and 188K database-tables (extracted from BI data models) [33], referred to as C𝑑𝑏, and synthesize table-tasks based on real tables sampled from the corpus.

We will go over the list of synthesized table-tasks below: (T-13) Table summarization (TS). Since web-tables often have descriptive titles, we synthesize a table-summarization task, where we ask the model to summarize the content in a table. Specifically, we sample 𝑇 ∈ C𝑤𝑡 whose extracted table-title 𝑡𝑖𝑡𝑙𝑒(𝑇) are neither too long nor too short, and create a table-summarization task as:

𝑇𝑆(𝑇) = (𝐼𝑛𝑠𝑇𝑆,𝑇,𝑡𝑖𝑡𝑙𝑒(𝑇))

where 𝐼𝑛𝑠𝑇𝑆 is the canonical human-instruction to describe the TS task (e.g., “Please provide a succinct summary for the table below”), which we will further augment for diversity (Section 4.3),𝑇 is the input table we sampled from C𝑤𝑡, and 𝑡𝑖𝑡𝑙𝑒(𝑇) is its expected completion.

This task is designed to use real tables, with real human annotated titles, to enhance models ability to read tables and understand the highlights from the table. Note that although we use 𝑡𝑖𝑡𝑙𝑒(𝑇) as the expected completion/answer, it does not overconstrain language-models to over-fit on such answers – it only nudges language-models in that general direction, just like training data in the form of (“write a bed-time story with a bear goes to beach” → an-actual-human-written-story) used in instructiontuning does not over-constrain/over-fit the underlying models.

(T-14) Column augmentation. Since we have lots of real tables in C𝑤𝑡 and C𝑑𝑏, we create a task where we take the first 𝑘 columns in a table 𝑇, denoted as 𝐶[1,𝑘](𝑇), and ask the language-models to generate the (𝑘 + 1)-th column 𝐶𝑘+1(𝑇), written as follows:

𝐶𝐴(𝑇,𝑘) = (𝐼𝑛𝑠𝐶𝐴,𝐶[1,𝑘](𝑇),𝐶𝑘+1(𝑇))

where 𝐼𝑛𝑠𝐶𝐴 is again the natural-language instruction that describes the row-augmentation task. This task exercises a model’s ability to generate realistic columns given a table context that need to be semantically compatible.

(T-15) Row augmentation (RA).SimilartoColumn-augmentation,

we synthesize a Row-augmentation task where we sample a table 𝑇 and ask the model to generate the (𝑘 + 1)-th row, given the first 𝑘 rows, written as:

𝑅𝐴(𝑇,𝑘) = (𝐼𝑛𝑠𝑅𝐴,𝑅[1,𝑘](𝑇),𝑅𝑘+1(𝑇))

This task exercises a model’s ability to synthesize realistic rows given a table context, which need to align vertically with existing rows.

(T-16) Row/column swapping (RS/CS). In this task, we ask the models to perform a table-manipulation step, where given a sampled table 𝑇, we provide an instruction to swap the 𝑖-th and 𝑗-th row. We programmatically generate the resulting output table from the swap operation, denoted as 𝑆𝑤𝑎𝑝(𝑇,𝑅𝑖,𝑅𝑗), which is the target “completion”. The Row-swapping task 𝑅𝑆𝑖,𝑗 (𝑇) is written as:

𝑅𝑆𝑖,𝑗 (𝑇) = (𝐼𝑛𝑠𝑅𝑆,𝑇,𝑆𝑤𝑎𝑝(𝑇,𝑅𝑖,𝑅𝑗)) We similarly synthesize the Column-swapping task 𝐶𝑆𝑖,𝑗 (𝑇) as: 𝐶𝑆𝑖,𝑗 (𝑇) = (𝐼𝑛𝑠𝐶𝑆,𝑇,𝑆𝑤𝑎𝑝(𝑇,𝐶𝑖,𝐶𝑗))

We note that tasks like Row/Column-swapping would seem simple to perform, both programmatically or through UI interactions (e.g., inside spreadsheets using menu options), and are therefore not tasks studied in the literature (unlike more challenging tasks like entitymatching or data-transformation). We are similarly not intending to use table tasks as “tests”, but because “tables serialized as naturallanguage texts” are ultimately the only way to feed input into language models (regardless of whether we want to output to be text/code/table/etc.), these table-tasks are still useful as “training data” for models to better read and understand tables.

(T-17) Row/column filtering. In this table-manipulation task, we ask models to filter down to specific rows/columns on a sampled

table 𝑇, based on a specified set of row/column indexes 𝑆:

𝑅𝐹𝑆 (𝑇) = (𝐼𝑛𝑠𝑅𝐹,𝑇,𝑅𝑆 (𝑇)) 𝐶𝐹𝑆 (𝑇) = (𝐼𝑛𝑠𝐶𝐹,𝑇,𝐶𝑆 (𝑇))

These tests are again meant to exercise model ability to manipulate tables, where cells in both vertical and horizontal directions need to be aligned.

(T-18) Row/column sorting (RS/CS). In the sorting tasks, we ask models to sort rows in a table 𝑇, based on values in a column 𝐶, where the expected output table can be programmatically generated, which we write as 𝑆𝑜𝑟𝑡𝐶(𝑇), so that the task 𝑅𝑆𝐶(𝑇) is:

𝑅𝑆𝐶(𝑇) = (𝐼𝑛𝑠𝑅𝑆,𝑇,𝑆𝑜𝑟𝑡𝐶(𝑇)) Similarly, we have a task to sort columns in a table 𝑇, based on column-headers 𝐻, written as 𝐶𝑆𝑠(𝑇):

𝐶𝑆(𝑇) = (𝐼𝑛𝑠𝐶𝑆,𝑇,𝑆𝑜𝑟𝑡𝐻 (𝑇)) We note that the sorting tasks are fairly challenging for languagemodels – while we do not expect models to be perfect on such tasks, they exercises model ability to manipulate tables nevertheless.

(T-11) Head-value matching (HVM). In this task, we sample a table𝑇, remove all its column headers 𝐻 to produce the corresponding table without headers, 𝑇. We then shuffle these headers 𝐻, and ask models to fill 𝐻 into𝑇′, to produce the HVM(𝑇) task:

HVM(𝑇) = (𝐼𝑛𝑠𝐻𝑉𝑀,𝑇,𝑇)

Like other tasks above, HVM is another task that we can synthesize in large quantities, using real tables, and without labeling. It is intended to be a task that helps models to better understand and correlate the semantics of column-headers and values.

Discussions. We show in our experiments, that using synthesized table-tasks on diverse tables improves the task- and data-diversity, which lead to better model generalizability (our ablation study shows that without these synthesized tasks there is a substantial drop in model quality).

Our list of synthesized table-tasks, however, is obviously not meant to be exhaustive, and is only a starting point. We believe that with some creativity, many more tasks can be synthesized to further improve the table-tuning process. For comparison, the NLP community has amassed over 1000 tasks for instruction-tuning, in a community effort [15], where they show that having more and diverse tasks always helps instruction-tuning.

Synthesize new table test-cases for data-diversity. There are a number of existing and important table-tasks, such as datatransformation, entity-matching, etc. that are extensively studied in the database literature. We want to use these established tasks in table-tuning too, also in the “(instruction, table, completion)” format. However, like mentioned earlier, the existing benchmarks for these tasks are typically manually labeled on a few datasets, which can be used to evaluation, but are unfit as training data for table-tuning, due to their limited quantities and diversity.

Instead, we synthesize new table test-cases for these established table-tasks, using real tables sampled from C𝑤𝑡 and C𝑑𝑏.

(T-5) Row-to-row Data Transformation (R2R) [23, 24]. To synthesize diverse test tables with data-transformations, we run a production-quality program-synthesizer [24], on web-tables sampled from C𝑤𝑏, to identify tables 𝑇 ∈ C𝑤𝑏 where some columns

𝐶𝑖𝑛 ⊂ 𝑇 can be transformed into 𝐶𝑜𝑢𝑡 ⊂ 𝑇, using an inferred program 𝑃, such that 𝑃(𝐶𝑖𝑛) = 𝐶𝑜𝑢𝑡 hold on all rows in 𝑇 (e.g., (first-name, last-name) → (full-name) in the same table [26]). We then remove one random value 𝑣 ∈ 𝐶𝑜𝑢𝑡 from 𝑇, to produce a test table𝑇−𝑣 where 𝑣 is missing. We then synthesize a task 𝑅2𝑅(𝑇):

𝑅2𝑅(𝑇) = (𝐼𝑛𝑠𝑅2𝑅,𝑇−𝑣,𝑇)

where given 𝑇−𝑣 as the input, we want to the model to infer the transformation and fill in the missing 𝑣 to produce𝑇.

(T-7) Schema Matching (SM) [44]. To synthesize new table test cases for schema matching, we sample a real table 𝑇, and take the first𝑘 rows of𝑇 to produce𝑇1 = 𝑅[1, 𝑘](𝑇). We then take the next𝑘 rows from 𝑇 to produce 𝑇2 = 𝑅[𝑘+1, 2𝑘](𝑇), where we additionally “paraphrase” the column-headers of the original𝑇, into new columnheaders in 𝑇2, using a mapping of semantically-similar columnnames generated by GPT, denoted as 𝑀 (e.g., “company names” → “enterprises”, “emp-id” → “employee identifier”, etc.). Finally, we shuffle the columns in 𝑇1 and 𝑇2, and make the two a test case for schema matching, where the ground-truth is in 𝑀. The resulting task is written as 𝑆𝑀(𝑇):

𝑆𝑀(𝑇) = (𝐼𝑛𝑠𝑆𝑀, (𝑇1,𝑇2),𝑀)

This again can systematically generate large numbers of schemamatching test tables, as training data for table-tuning.

(T-8) Data Imputation (DI) [7, 37]. For data imputation, we randomly sample a real table𝑇, and then remove a random value 𝑣 ∈ 𝑇, to produce 𝑇−𝑣. The task 𝐷𝐼(𝑇) is then to predict the missing 𝑣 from its table context:

𝐷𝐼(𝑇) = (𝐼𝑛𝑠𝐷𝐼,𝑇−𝑣,𝑣)

Note that while not all missing values 𝑣 in DI tasks so generated can be reliably predicted, it nevertheless exercises models’ ability to leverage correlations that exist between values in the row and column contexts.

(T-9) Error Detection (ED) [45]. To synthesize error-detection tasks, we sample a real table 𝑇 ∈ C𝑤𝑡, and generate a modified 𝑇˜, where we replace a value 𝑣 ∈ 𝑇 with 𝑣′, using an existing package [3] that injects one likely typographic error into 𝑣. The task 𝐸𝐷(𝑇) is then:

𝐸𝐷(𝑇) = (𝐼𝑛𝑠𝐸𝐷,𝑇˜,𝑣′)

where we aim to identify the misspelled 𝑣′ ∈ 𝑇˜ based on surrounding table context.

(T-10) List extraction (LE) [13, 19]. To synthesize the task of extracting tables from list data without explicit column-delimiters, we sample a table 𝑇, and replace all column separators with white spaces to generate its unsegmented list-form 𝐿(𝑇). The task 𝐿𝐸(𝑇) is then:

𝐿𝐸(𝑇) = (𝐼𝑛𝑠𝐿𝐸,𝐿(𝑇),𝑇)

which is to produce the correct column-segmentation of 𝐿(𝑇), and generate the corresponding table 𝑇, based on value alignment in the vertical direction.

Since we have large numbers of diverse tables, in Line 3 of Algorithm 1 we make sure that each table𝑇 is used by one task-type above, to synthesize one instance of table-task, to ensure the diversity of data we generate.

[Figure 7]

- Figure 7: Example table-tasks we generate for (T-8) Entitymatching, and (T-9) Error-detection, using “augmentedcompletions” that contain reasoning steps similar to chainof-thought, which when used as training-data in table-tuning, can ground model responses and improve result quality.

### 4.3 Augment synthesized table-tasks

From synthesized instances of table-tasks 𝑡 = (𝐼𝑛𝑠,𝑇,𝐶), we then perform additional augmentations at multiple levels, corresponding to steps in Line 6-Line 8 of Algorithm 1, where the goal is to create even more task/data diversity and avoid over-fitting in table-tuning.

We will go over different levels of augmentations below in turn. Instruction-level augmentations. At the instruction level,

because using the same instruction repeatedly across training-data instances can lead to over-fitting [57], we augment the canonical instruction using generative models like GPT to paraphrase the canonical human-written instruction into many different variants.

For example, for the task-type (T-13): Table-Summarization (Section 4.2), the canonical human-written instruction is: “Please look

- at the table below and provide a title that can summarize the table”. We generate alternative instructions for the task using language-models, to produce variations such as “Please examine the table below and give it a descriptive title”, in a manner similar to [57], which we can then use to populate instances of table-tasks as alternative instructions (Line 6).

Table-level augmentations. At the table-level, we know that two-dimensional tables should largely be “permutation-invariant”, where permutating rows and columns should generally lead to a table with similar semantic meanings (Section 3), at the tablelevel we can perform operations such as column-permutation, rowpermutation, column-sampling, and row-sampling, to increase the diversity of tables used in our table tasks.

When the training data has an original instance of the table-task, 𝑡 = (𝐼𝑛𝑠,𝑇,𝐶), and its augmented version𝑡′ = (𝐼𝑛𝑠,𝑇′,𝐶), where𝑇′ is an augmented version of𝑇, which has the same semantic meaning and thus the same completion 𝐶, the hope is that by continuing to train language-models on such training-data, we can increase model stability on tables and make them less sensitive to “semanticpreserving table-operations” (e.g., column-reordering like discussed in Section 3).

Completion-level augmentations. At the completion-level, for a synthesized instance of table-task 𝑡 = (𝐼𝑛𝑠,𝑇,𝐶), we augment our original completion 𝐶 by generating 𝐶′ that adds reasoning steps into the original 𝐶, after we observe that performing reasoning-steps (similar to [60]) on more complex table-tasks (e.g., entity-matching and error-detection) can lead to better task performance.

Language-model assisted completion-augmentation.We showthe

completion-level augmentation using examples in Figure 7. The left of the figure is the task of (T-8) Entity-matching, where models need to identify whether two table-rows may refer to the same real-world entities. We tried two modes of operations on vanilla language-models (before table-tuning):

- (1) Direct-answer: where models are asked to produce a “yes/no” answer directly, without additional reasoning;
- (2) Reason-then-answer: where we require the models to “explain step-by-step” in the prompt, before producing a “yes/no” answer.

We observe that for complex tasks like Entity-matching, models in the “direct-answer” mode are more prone to make mistakes (e.g., answering with only a “no” in the response), leading to lower accuracy; whereas with “reason-then-answer”, even vanilla language-models perform better, likely because the step-by-step reasoning chain makes it likely to generate the correct answer, similar to what was observed in the NLP literature (e.g., chain-of-thought reasoning [60] and self-consistency [55], etc.).

Recall that our goal is to table-tune language models, so that the resulting model can perform better on complex table-tasks, by hopefully developing a habit of performing step-by-step reasoning. In order to produce training-data for table-tuning, so that languagemodels can learn-to-reason on complex tasks, we augment 𝑡 = (𝐼𝑛𝑠,𝑇,𝐶) by replacing the direct yes/no answer 𝐶, with a more detailed answer using step-by-step reasoning (Figure 7 shows a simplified example), which we generate by providing GPT with the yes/no answer, and then ask the model to produce reasoning-steps as our new 𝐶′. We find this to encourage table-tuned models to perform correct reasoning-steps, on other complex table-tasks that are similar in nature.

Ground-truth assisted completion-augmentation. The right of

Figure7showsanotherexampleofcompletion-augmentation, where we use ground-truth to produce step-by-step reasoning, for the example task of (T-9): Error-detection. The task here is to identify misspelled cells from a given table. We find that vanilla languagemodels to produce lots of false-positives on the error-detection task, where the models confidently predict a cell-value (e.g., an abbreviation or uncommon company/person/location names) to be misspelled, when no such misspellings exist upon close inspection. The tendency to produce false-positives persists, with or without “step-by-step” chain-of-thought style prompting.

Because we cannot rely on chain-of-thought style prompts to generate correct reasoning steps as augmented-completions 𝐶′, we instead use the ground-truth available to us in the table-task synthesis step (Section 4.2), to generate augmented completions that embeds the reasoning step. Specifically, like shown in Figure 7 (right), we augment the completion to contain not only the prediction (a value 𝑣 is misspelled), but also generate the explanation that points out the correct version of the predicted misspelling

(e.g., “Missisipi” should be “Mississippi”). We find this grounds the language-models predictions with an actual explanation, which substantially reduces false-positives and improve result quality.

Additional augmentations. Along the lines of augmentations, there are additional types of augmentations we perform, including “template-level augmentation”, where we mix zero-shot task template and few-shot task template (which appends multiple inputtable/output-completion examples after the instruction 𝐼𝑛𝑠), as well as “task-level augmentation” (by synthesizing new types of tabletasks), which all improve training-data diversity and help tabletuning.

### 4.4 Table-GPT as “table foundation models”

Using the synthesis-then-augment approach in Algorithm 1, describe in previous sections, we now generate large numbers of diverse table-tasks 𝐴 = {(𝐼𝑛𝑠,𝑇,𝐶)}. We then continue to train language models such as GPT, using serialized (𝐼𝑛𝑠,𝑇) as the “prompt”, and𝐶 as the“completion”,whereweminimize the language-modeling loss of completion given the prompt, subject to regularization. We refer to this process as table-tuning.

Let 𝑀 be a decoder-style language model, such as GPT and ChatGPT, let TableTune(𝑀) be the table-tuned version of 𝑀. We argue that TableTune(𝑀) could serve as a better “table foundation model”, if it performs better than 𝑀 on table-tasks, in all of the following scenarios:

- (1) Out of the box zero-shot: when we use only instructions for 𝑀 or TableTune(𝑀) to perform table-tasks;

- (2) Out of the box few-shot: when we use instructions and randomly selected few-shot examples to perform table-tasks;

- (3)Task-specific prompt-tuning: whenwehavea small amountof labeled data for a downstream task, and perform prompt-tuning to select the best instruction/example combinations;

- (4) Task-specific fine-tuning: when we have sufficient amounts of labeled data, and perform task-specific fine-tuning for a task.

If table-tuning is effective for language models to learn to better understand and manipulate tables, we expect that TableTune(𝑀) can perform better on most if not all of the scenarios described above, which is the goal of our experimental evaluation next.

- 5 EXPERIMENTS

We perform extensive experiments to evaluate table-tuned GPT relative to vanilla GPT on diverse table tasks. We plan to release our code and data after internal reviews1 .

- 5.1 Experiment Setup Models Compared. We test the following models.

- • GPT-3.5 (text-davinci-002). This 175B model is available from OpenAI, and is one of the vanilla GPT models that we compare with.

- • Table-GPT-3.5 (text-davinci-002 +table-tune). Thisisthemodelwe obtain by performing table-tuning on GPT-3.5 (text-davinci-002). We compare the performance of Table-GPT-3.5 with GPT-3.5.

- • ChatGPT (text-chat-davinci-002). This is a version of the ChatGPT model available internally [28], which we use as a second vanilla base model, from which we perform table-tuning.

1: https://aka.ms/table-gpt

• Table-ChatGPT (text-chat-davinci-002 +table-tune). Thisisthemodel we obtain by performing table-tuning on ChatGPT (text-chatdavinci-002), which we compare with the vanilla ChatGPT.

Training tasks and data. In our default settings, we use a total of 14 types of table-tasks, listed as T-5 to T-18 in Table 2, as training data for table-tuning.

In all but two task-types (T-6: Entity Matching and T-12: NL-toSQL), we use synthesized instances of table-tasks. For each task type, we generate 1000 instances of table-tasks using a 50:50 mix of zero-shot and few-shot templates, following a synthesis-thenaugment approach described in Section 4. During task-synthesis, we sample from 125k real web-tables C𝑤𝑡 and database-tables C𝑑𝑏 (aggressively deduped from over 2M original tables). For Entity Matching and NL-to-SQL where realistic labels/completions cannot be automatically synthesized, we use existing manually-labeled benchmark data, from [1] and [65], respectively.

Test tasks and data. To evaluate the benefit of table-tuning, we test the performance of paired models that are table-tuned vs. vanilla un-tuned, namely, we compare (GPT-3.5 vs. Table-GPT-3.5) and (ChatGPT vs. Table-ChatGPT), as two groups.

We test on 4 unseen tasks (T-1 to T-4 in Table 2), which are completely unseen during table-tuning, to evaluate whether our table-tuned models can continue to follow to human-instructions and perform novel unseen tasks (like illustrated in Figure 4). In addition, we make sure that the test-data used in unseen tasks, are completely separate from the tables used in synthesizing table-tasks as training-data for table-tuning. Specifically, our training data for table-tuning are always drawn from web-tables C𝑤𝑡 and databasetables C𝑑𝑏, whereas test-data used in our synthesized table-tasks (T1: Missing-value identification and T2: Column-finding) are always drawn from a corpus of real spreadsheet tables C𝑠𝑝, completely separate from C𝑤𝑡 and C𝑑𝑏 and with very different characteristics. For the remaining two unseen tests (T-3: Table Question and T-4: Column Type Annotation), we use established benchmark data [42] and [16, 25, 51] respectively, which are unseen during table-tuning.

We also evaluate 5 seen tasks (T-5 to T-9 in Table 2), which are important table-tasks extensively studied in the literature, which we want table-tuned models to be exposed of to understand these table-related concepts. While these task-types are seen during tabletuning, we make sure that the test datasets are completely separate from the training data used in table-tuning. For synthesized tabletasks (T-8 Data Imputation), similar to discussed above, our test cases are always drawn from a corpus of real spreadsheet tables C𝑠𝑝, separate from the corpus of web-tables C𝑤𝑡 and database-tables C𝑑𝑏 used in synthesizing training table-tasks, in order to test tabletuned models’ ability to generalize to new tables. For other tasks, we use existing benchmark data, completely unseen when training table-tuned models (e.g., [24] for T-5: Row-to-row transformation, [1] for T-6: Entity-matching, using the same setup as [39], [30] for T-7: Schema-matching). The task of (T-9) Error-detection is of high value for our business, where we manually labeled a benchmark using real spreadsheet-tables and web-tables for this evaluation.

Details of test data and their statistics can be found in Table 4.

|Task Type<br><br>|Task|Dataset| |Zero-Shot GPT-3.5 +table-tune| | |Few-Shot GPT-3.5 +table-tune| | |Zero-Shot ChatGPT +table-tune| | |Few-Shot ChatGPT +table-tune| | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | |
|Unseen|Column Finding|Spreadsheets-CF Efthymiou| |0.461 0.757<br><br>|0.713 0.886| |0.682 0.784<br><br>|0.816 0.847| |0.699 0.823<br><br>|0.807 0.882| |0.803 0.806<br><br>|0.848 0.861| |
| |Column Type Annotation<br><br>|Limaye| |0.683|0.755| |0.719|0.853| |0.742|0.769| |0.832|0.853| |
| | | | | | | | | | | | | | | | |
| | |Sherlock| |0.332|0.449| |0.528|0.538| |0.454<br><br>|0.482| |0.521<br><br>|0.553| |
| | |T2D| |0.776|0.875| |0.83<br><br>|0.915| |0.827<br><br>|0.886| |0.853<br><br>|0.912| |
| |Missing Value Identification<br><br>|Column (no separator) Column (with separator)| |0.261 0.305<br><br>|0.294 0.457| |0.383 0.519<br><br>|0.441 0.643| |0.299 0.422<br><br>|0.351 0.520| |0.468 0.635|0.474 0.665| |
| | | | | | | | | | | | | | | | |
| | |Row (no separator)| |0.768<br><br>|0.851| |0.774<br><br>|0.882| |0.822<br><br>|0.840| |0.859<br><br>|0.894| |
| | |Row (with separator)| |0.875<br><br>|0.959| |0.917|0.976| |0.923<br><br>|0.936| |0.960|0.968| |
| |Table Question<br><br>|Wiki| |0.45|0.486| |0.454|0.478| |0.512<br><br>|0.521| |0.520<br><br>|0.527| |
|Seen|Data Imputation<br><br>|Spreadsheets-DI Amazon-Google| |0.423 0.153|0.558 0.657| |0.57 0.659<br><br>|0.625 0.676| |0.524 0.239|0.594 0.566| |0.609 0.680<br><br>|0.649 0.701| |
| |Entity Matching<br><br>|Beer| |0.5|0.727| |0.815|0.923| |0.741|0.923| |0.783|0.963| |
| | | | | | | | | | | | | | | | |
| | |DBLP-ACM| |0.402|0.847| |0.954|0.912| |0.833<br><br>|0.932| |0.961|0.938| |
| | |DBLP-GoogleScholar| |0.206<br><br>|0.861| |0.809<br><br>|0.896| |0.632|0.912| |0.823|0.924| |
| | |Fodors-Zagats| |0.083|0.872| |0.872<br><br>|0.977| |0.809|1.000| |0.872<br><br>|0.977| |
| | |Walmart-Amazon| |0.268|0.691| |0.519|0.711| |0.206<br><br>|0.678| |0.664|0.824| |
| | |iTunes-Amazon| |0|0.788| |0.826<br><br>|0.943| |0.393|0.862| |0.833<br><br>|0.929| |
| |Error Detection|Spreadsheets-Real WebTables-Real| |0.058 0.077|0.565 0.643| |0.319 0.338<br><br>|0.552 0.545| |0.058 0.078|0.544 0.656| |0.443 0.364<br><br>|0.551 0.684| |
| | | | | | | | | | | | | | | | |
| |Schema Matching<br><br>|DeepM| |1|1| |1<br><br>|1| |0.857<br><br>|1| |1|1| |
| |Row-to-Row Transformation<br><br>|BingQL-Unit BingQL-other| |N.A.<br><br>| | |0.213 0.431<br><br>|0.427 0.588| |N.A.| | |0.339 0.558<br><br>|0.446 0.607| |
| | | | | | | | | | | | | | | | |
| | |FF-GR-Trifacta| | | | |0.712|0.788| | | | |0.772<br><br>|0.825| |
| | |Headcase| | | | |0.636<br><br>|0.705| | | | |0.704|0.795| |
| | |Stackoverflow| | | | |0.662|0.745| | | | |0.800<br><br>|0.758| |

#### Table 3: Detailed results of Table-tuning, on both GPT-3.5 and ChatGPT, for individual datasets. Zero-shot is not applicable to row-to-row by-example transformations (marked as “N.A.”), which requires examples. For all “Unseen” tasks, the tasks are held-out and unseen during table-tuning. For all “Seen” tasks, the task is seen during table-tuning, but the test datasets are held-out and unseen.

0.604

0.863

0.625

0.549

0.777 0.779

0.702

0.558 0.570

0.328

0.603

0.067

0.423

0.230

|Entity Matching Error Detection Data Imputation Row-to-Row Transformation<br><br>0.637<br><br>0.741<br><br>0.715<br><br>0.788<br><br>Column Type Annotation<br><br>0.552<br><br>0.640 0.648<br><br>0.736<br><br>Missing Value Identification<br><br>0.461<br><br>0.713 0.682<br><br>0.816<br><br>Column Finding<br><br>0.450<br><br>0.486<br><br>0.454<br><br>0.478<br><br>Table Question<br><br>0.423<br><br>0.230<br><br>0.057<br><br>1.000<br><br>0.450<br><br>0.558<br><br>0.777<br><br>0.628<br><br>1.000<br><br>0.486<br><br>0.570<br><br>0.779<br><br>0.357<br><br>1.000<br><br>0.454<br><br>0.625<br><br>0.863<br><br>0.637<br><br>1.000<br><br>0.478<br><br>Data Imputation Entity Matching Error Detection Schema Matching Table Question<br><br>GPT-3.5 Zero-Shot Table-GPT-3.5 Zero-Shot GPT-3.5 Few-Shot Table-GPT-3.5 Few-Shot<br><br>|
|---|

#### Figure 8: Overall quality improvement, between vanilla GPT-3.5 and Table-GPT-3.5.

0.649

0.893

0.618

0.600

0.594 0.609

0.839

0.734

0.802

0.404

0.524

0.682

0.550

0.068

|Entity Matching Error Detection Data Imputation Row-to-Row Transformation<br><br>0.712<br><br>0.755 0.753<br><br>0.795<br><br>Column Type Annotation<br><br>0.617<br><br>0.662<br><br>0.731 0.750<br><br>Missing Value Identification<br><br>0.699<br><br>0.807 0.803<br><br>0.848<br><br>Column Finding<br><br>0.513<br><br>0.521 0.520<br><br>0.527<br><br>Table Question<br><br>0.524 0.550<br><br>0.052<br><br>0.857 0.594 0.513 0.839<br><br>0.636<br><br>1.000<br><br>0.521<br><br>0.609<br><br>0.802<br><br>0.436<br><br>1.000<br><br>0.520<br><br>0.649<br><br>0.893<br><br>0.693<br><br>1.000<br><br>0.527<br><br>Data Imputation Entity Matching Error Detection<br><br>|Schema|
|---|
<br><br>a Matching<br><br>|Table|
|---|
<br><br>Ta Question<br><br>ChatGPT Zero-Shot Table-ChatGPT Zero-Shot ChatGPT Few-Shot Table-ChatGPT Few-Shot<br><br>|
|---|

#### Figure 9: Overall quality improvement, between vanilla ChatGPT and Table-ChatGPT.

Table 4: Details of test data and evaluation metrics

labeled examples. We will study the benefit of table-tuning in these two settings below.

|Task<br><br>|Evaluation Metrics|Datasets<br><br>|Size|
|---|---|---|---|
|T-1: Missing Value Identification<br><br>|F1|Column (no Separator) Column (with Separator)<br><br>|1000 1000|
| | | | |
| | |Row (no Separator)<br><br>|1000|
| | |Row (with Separator)|1000|
|T-2: Column Finding|Acc<br><br>|Spreadsheets-CF<br><br>|841|
|T-3: Table Question<br><br>|Acc|Wiki|4344|
|T-4: Column Type Annotation<br><br>|F1<br><br>|Efthymiou Limaye<br><br>|594 174|
| | | | |
| | |Sherlock<br><br>|971|
| | |T2D|367|
|T-5: Row-to-Row Transformation|Acc<br><br>|BingQL-Unit BingQL-other<br><br>|103 1102|
| | | | |
| | |FF-GR-Trifacta|132|
| | |Headcase<br><br>|88|
| | |Stackoverflow<br><br>|145|
|T-6: Entity Matching<br><br>|F1|Amazon-Google Beer<br><br>|2293 91|
| | | | |
| | |DBLP-ACM<br><br>|2473|
| | |DBLP-GoogleScholar<br><br>|5742|
| | |Fodors-Zagats<br><br>|189|
| | |Walmart-Amazon<br><br>|2049|
| | |iTunes-Amazon|109|
|T-7: Schema Matching<br><br>|Recall<br><br>|DeepM<br><br>|41|
|T-8: Data Impuation<br><br>|Acc<br><br>|Spreadsheets-DI<br><br>|1000|
|T-9: Error Detection<br><br>|F1|Spreadsheets-Real WebTables-Real<br><br>|870 432|
| | | | |

#### Single-taskprompt-engineering: Weperformprompt-engineering

for Table-GPT-3.5 and GPT-3.5, on the column-type-annotation (CTA) task (using the Efthymiou [16] dataset), by selecting the best few-shot examples using 200 labeled examples (randomly sampled from the ground-truth), where the goodness of a prompt is evaluated on the labeled examples. Figure 10 shows the top-5 prompts selected, for Table-GPT-3.5 and GPT-3.5, respectively. We can see that Table-GPT-3.5 consistently outperforms GPT-3.5, on the 5 best prompts produced from prompt-engineering.

Single-task fine-tuning: We perform task-specific fine-tuning, on Table-GPT-3.5 and GPT-3.5, using labeled data for that specific task. Table 11(a) and Table 11(b) show the comparison, on the CTA task (using Efthymiou [16]) and Table-Question-Answering or TQA (using WikiTableQuestions [42]), respectively. In both cases, we vary the amount of training data on the x-axis. As expected, the performance of both Table-GPT-3.5 and GPT-3.5 benefit from continued task-specific fine-tuning, but with the same amount of training data, Table-GPT-3.5 continues to dominate GPT-3.5. Looking at the graph from a different way, to achieve the same performance (y-axis), fine-tuning Table-GPT-3.5 would require a smaller number of labeled data than fine-tuning the vanilla GPT-3.5.

### 5.2 Quality Comparisons: Unseen + Seen tasks

In Figure 8, we compare the performance between (GPT-3.5 vs. Table-GPT-3.5), and in Figure 9, we compare the performance between (ChatGPT vs. Table-ChatGPT), which are table-tuned vs. untuned vanilla models, using GPT-3.5 and ChatGPT as base-models, respectively. Within each task-group in the figures, we show 4 bars, where the first two correspond to zero-shot settings, and the last two correspond to few-shot settings. We can see that across the board, table-tuned models show strong performance benefits on diverse table-tasks.

0.90

|0.822<br><br>0.785<br><br>0.816<br><br>0.801 0.799<br><br>0.877<br><br>0.867<br><br>0.851<br><br>0.870<br><br>0.860|
|---|

AveragePerformance

0.85

0.80

0.75

1 2 3 4 5

It is interesting to note that the benefit of table-tuning is observed when both GPT-3.5 and ChatGPT are used as base-models, showing the generality of our proposed table-tuning approach, on top of underlying language-models of different styles.

Top 5 Prompt Template GPT-3.5 Table-GPT-3.5

#### Figure 10: Comparison of quality, when using promptengineering. Results shown are for 5 best prompt-templates on the Efthymiou dataset.

Table 4 shows a detailed breakdown of the results, at the individual data-set level. We can see that across 26 test datasets, on 2 base-models (GPT-3.5 and ChatGPT), in 2 settings (zero-shot and few-shot), for a total of 104 tests, table-tuned models outperform their vanilla un-tuned counterparts in 98/104 tests (with the remaining being 3 ties and 3 losses), showing the strong performance benefits of table-tuning.

1

0.6

AveragePerformance

AveragePerformance

### 5.3 Benefits on task-specific optimizations

0.55

0.9

In addition to performing well out-of-the-box in zero-shot and (random) few-shot settings, as shown above, table-tuned GPT models could potentially be used as “table foundation models”, if they continue to show quality benefits on downstream tasks, when taskspecific optimizations are applied.

0.5

0.8

0.45

0.7

0.4

0 100 200 400

0 100 200 400 800

Size of Training Data GPT-3.5 Table-GPT-3.5

Size of Training Data GPT-3.5 Table-GPT-3.5

Like we discussed in Section 4.4, these include (1) single-task prompt-engineering, where we select the best instructions and fewshot examples for a single task, using a small number of labeled examples; and (2) single-task fine-tuning, where we continue to fine-tune models for a specific task, with a sufficient number of

(a) CTA (Efthymiou)

(b) TQA (Wiki)

#### Figure 11: Single Task Fine-tuning

Table-GPT-3.5 and GPT-3.5, but the former consistently outperforms the latter by more than 10 percentage points on all 5 prompt templates, showing the robustness of Table-GPT to different kinds of prompts.

CF CTA DI ED MVI R2R SM TQA

AveragePerformanceScore

Vary table formats. There are multiple options when serializing a table 𝑇 into text, such as Markdown, CSV, JSON, etc. We use the Markdown table format, because it is succinct, and furthermore vanilla GPT tends to generate tables in the Markdown format in responding to human questions, suggesting that it is the table format of its choice, likely because GPT is pre-trained on lots of GitHub code, where Markdown tables are abundant. To understand the effect of using different table formats in representing tables in prompts, we test two different table formats, namely CSV and JSON.

Table-GPT-3.5

GPT-3.5

- 0

0.2

0.4

0.6

- 0.8
- 1

- 1 task 5 tasks 10 tasks All Tasks

Number of Training Tasks

#### Figure 12: Vary number of training tasks

- 5.4 Sensitivity Analysis We perform sensitivity analysis to better understand table-tuning.

Table 5 shows the average performance when using different table formats. As we can see, the Markdown format on average performs better than other formats, although the gap is not too significant.

Varying the number of training tasks. To see whether using more training tasks brings a general benefit, we sample 1/5/10 tasks from all of our training table-tasks for 4 times each, perform finetuning on each subset of tasks selected, and compute the average from these runs.

Table 5: Performance of Table-GPT-3.5, when different table formats are used to serialize tables

Task Type Markdown CSV JSON

The average quality results are shown in Figure 13. As we can see, on the left of the figure with a small number of tasks (e.g., 1), table-tuning degenerates to single-task tuning, which actually hurts the performance of other tasks in general (notice that the performance corresponding to 1-task is lower than the dotted green line, which corresponds to GPT-3.5). As we have more training-tasks, the performance goes up consistently, for all tasks as well as for the average across all tasks, showing the benefit that is analogous to multi-task training.

Seen 0.739 0.707 0.713 Unseen 0.663 0.662 0.621 Overall 0.705 0.687 0.672

### 5.5 Ablation Studies

We perform ablation analysis to understand the benefit of different augmentation strategies (Section 4.3). The results are summarized in Table 6.

Vary the amount of training data. Figure 13 shows the average performance on seen/unseen test tasks with different amounts of training data. As we can see, the performance on both seen and unseen tasks improves with training data, which plateaus as more data is used.

#### Table 6: Ablation Studies of table-tuning

###### Task Type GPT-3.5 Table-GPT-3.5 NoSyn NoColPer. NoPromptVar. NoCOT

Seen 0.548 0.739 0.610 0.735 0.722 0.728 Unseen 0.547 0.663 0.607 0.661 0.657 0.666 Overall 0.548 0.705 0.608 0.702 0.693 0.701

Vary base-model Size. To understand how the size of the basemodels affects the performance of table-tuned models, we tabletune four variants of GPT, namely, Text-Ada-001 (350M parameters), Text-Babbage-001 (3B parameters), Text-Curie-001 (13B parameters), Text-Davinci-002 (175B parameters).

No task-level augmentation (no synthesized tasks). Because we synthesized diverse table-tasks for table-tuning (Section 4.2), our first ablation is to remove all such tasks from the training data. The result is shown in Table 6 as “NoSyn". As we can see, the average performance on seen and unseen tasks drops significantly, showing the contribution of our diverse synthesized table-tasks.

Figure 14 shows the average performance of base-models vs. table-tuned models, on seen/unseen tasks. We can see that for the unseen tasks (important to check for model generalizability), tabletuned models produce little benefit on smaller models (Ada/Babbage/Curie), but the benefit becomes suddenly significant on larger models (GPT-3.5 and ChatGPT), which appear to be an emerging ability consistent with what is reported in other contexts (e.g., [8, 59]).

No table-level augmentation (no column permutations). We disable the table-level augmentation by turning off the column permutation. The result is shown in Table 6 as “NoColPer". We can see that the average performance on seen and unseen tasks is lowered without column permutations.

Vary prompt templates. To test the robustness of our table-tuned models, we generate 5 different prompt templates (task descriptions and special markers), which are paraphrased automatically using GPT, from a canonical prompt template written by humans.

No instruction-level augmentation (no prompt variations). We then disable the instruction-level augmentation, by using only one canonical prompt template for each task (without paraphrasing). The result is shown in Table 6 as “NoPromptVar". As we can see, the average performance of seen and unseen tasks drops slightly, likely because diverse types of table-tasks we use can somewhat mitigate the negative effect of using repeated instruction templates.

Figure 15 shows the average model performance over all unseen test tasks for each prompt template. As we can see, different prompt templates introduce variations in performance, for both

0.70

0.8

|0.548 0.552 0.548 0.556<br><br>0.543<br><br>0.652 0.657 0.662 0.668<br><br>0.654|
|---|

0.8

AveragePerformance

AveragePerformance

AveragePerformance

0.6

0.65

0.7

0.4

0.60

0.6

0.2

0.55

0.5

0.0

0.50

12.5% 25% 50% 100%

Ada (350M) Babbage (3B) Curie (13B) Davinci (175B)

1 2 3 4 5

Size of Training Data GPT-3.5 Seen GPT-3.5 Unseen

Prompt Template

Model Name GPT-3.5 Seen GPT-3.5 Unseen

GPT-3.5 Table-GPT-3.5

Table-GPT-3.5 Seen Table-GPT-3.5 Unseen

Table-GPT-3.5 Seen Table-GPT-3.5 Unseen

Figure 13: Vary Training Size

Figure 14: Vary Model Size

#### Figure 15: Vary Templates

No completion-level augmentation (no chain-of-thought). We drop the augmentation at completion level by removing the chain-of-thought (COT) reasoning from the completion in the finetuning data. The result is shown in Table 6 as “NoCOT". The average performance on seen tasks becomes lower with no COT, which is expected.

- [14] Xu Chu, Ihab F Ilyas, Sanjay Krishnan, and Jiannan Wang. 2016. Data cleaning: Overview and emerging challenges. In Proceedings of the 2016 international conference on management of data. 2201–2206.
- [15] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. 2022. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416 (2022).
- [16] Xiang Deng, Huan Sun, Alyssa Lees, You Wu, and Cong Yu. 2022. Turl: Table understanding through representation learning. ACM SIGMOD Record 51, 1

(2022), 33–40.

- [17] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018).
- [18] Till Döhmen, Hannes Mühleisen, and Peter Boncz. 2017. Multi-hypothesis CSV parsing. In Proceedings of the 29th International Conference on Scientific and Statistical Database Management. 1–12.
- [19] Hazem Elmeleegy, Jayant Madhavan, and Alon Halevy. 2009. Harvesting relational tables from lists on the web. Proceedings of the VLDB Endowment 2, 1

(2009), 1078–1089.

- [20] Raul Castro Fernandez, Aaron J Elmore, Michael J Franklin, Sanjay Krishnan, and Chenhao Tan. 2023. How Large Language Models Will Disrupt Data Management. Proceedings of the VLDB Endowment 16, 11 (2023), 3302–3309.
- [21] Tianyu Gao, Adam Fisch, and Danqi Chen. 2020. Making pre-trained language models better few-shot learners. arXiv preprint arXiv:2012.15723 (2020).
- [22] Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A Smith. 2020. Don’t stop pretraining: Adapt language models to domains and tasks. arXiv preprint arXiv:2004.10964 (2020).
- [23] William R Harris and Sumit Gulwani. 2011. Spreadsheet table transformations from examples. ACM SIGPLAN Notices 46, 6 (2011), 317–328.
- [24] Yeye He, Xu Chu, Kris Ganjam, Yudian Zheng, Vivek Narasayya, and Surajit Chaudhuri. 2018. Transform-data-by-example (TDE) an extensible search engine for data transformations. Proceedings of the VLDB Endowment 11, 10 (2018), 1165–1177.
- [25] Madelon Hulsebos, Kevin Hu, Michiel Bakker, Emanuel Zgraggen, Arvind Satyanarayan, Tim Kraska, Çagatay Demiralp, and César Hidalgo. 2019. Sherlock: A deep learning approach to semantic data type detection. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1500–1508.
- [26] Zhongjun Jin, Yeye He, and Surajit Chauduri. 2020. Auto-transform: learningto-transform by patterns. Proceedings of the VLDB Endowment 13, 12 (2020), 2368–2381.
- [27] Sean Kandel, Andreas Paepcke, Joseph Hellerstein, and Jeffrey Heer. 2011. Wrangler: Interactive visual specification of data transformation scripts. In Proceedings of the sigchi conference on human factors in computing systems. 3363–3372.
- [28] Tom Kocmi and Christian Federmann. 2023. Large language models are state-ofthe-art evaluators of translation quality. arXiv preprint arXiv:2302.14520 (2023).
- [29] Keti Korini and Christian Bizer. 2023. Column Type Annotation using ChatGPT. arXiv preprint arXiv:2306.00745 (2023).
- [30] Christos Koutras, George Siachamis, Andra Ionescu, Kyriakos Psarakis, Jerry Brons, Marios Fragkoulis, Christoph Lofi, Angela Bonifati, and Asterios Katsifodimos. 2021. Valentine: Evaluating matching techniques for dataset discovery. In 2021 IEEE 37th International Conference on Data Engineering (ICDE). IEEE, 468–479.
- [31] Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Luke Zettlemoyer, Omer Levy, Jason Weston, and Mike Lewis. 2023. Self-Alignment with Instruction Backtranslation. arXiv preprint arXiv:2308.06259 (2023).
- [32] Yuliang Li, Jinfeng Li, Yoshihiko Suhara, AnHai Doan, and Wang-Chiew Tan.

2020. Deep entity matching with pre-trained language models. arXiv preprint arXiv:2004.00584 (2020).

- [33] Yiming Lin, Yeye He, and Surajit Chaudhuri. 2023. Auto-BI: Automatically Build BI-Models Leveraging Local Join Prediction and Global Schema Graph. arXiv preprint arXiv:2306.12515 (2023).

### 6 CONCLUSIONS AND FUTURE WORK

In this work, we propose a new paradigm called table-tuning, that can continue to fine-tune the model weights of pre-trained large language-models like GPT-3.5 and ChatGPT, such that the resulting models are better in understanding tables and performing table tasks, while still being versatile in following diverse human instructions for unseen tasks. Just like how instruction-tuning has turned into a rich and fruitful line of research in the NLP literature, we hope our initial steps in table-tuning can serve as a springboard for others to continue in this path to develop more optimized models for tables and table-related tasks.

### REFERENCES

- [1] [n.d.]. Magellan data repository. https://sites.google.com/site/anhaidgroup/ useful-stuff/the-magellan-data-repository?authuser=0.
- [2] [n.d.]. OpenAI: ChatGPT. https://openai.com/blog/chatgpt.
- [3] [n.d.]. Python typo generator. https://pypi.org/project/typo/.
- [4] [n.d.]. Stanford Alpaca. https://github.com/tatsu-lab/stanford_alpaca.
- [5] Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. 2023. Palm 2 technical report. arXiv preprint arXiv:2305.10403 (2023).
- [6] Simran Arora, Avanika Narayan, Mayee F Chen, Laurel J Orr, Neel Guha, Kush Bhatia, Ines Chami, Frederic Sala, and Christopher Ré. 2022. Ask me anything: A simple strategy for prompting language models. arXiv preprint arXiv:2210.02441

(2022).

- [7] Felix Biessmann, Tammo Rukat, Philipp Schmidt, Prathik Naidu, Sebastian Schelter, Andrey Taptunov, Dustin Lange, and David Salinas. 2019. DataWig: Missing Value Imputation for Tables. J. Mach. Learn. Res. 20, 175 (2019), 1–6.
- [8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.
- [9] Michael J Cafarella, Alon Y Halevy, Yang Zhang, Daisy Zhe Wang, and Eugene Wu. 2008. Uncovering the Relational Web.. In WebDB. Citeseer, 1–6.
- [10] Kaushik Chakrabarti, Surajit Chaudhuri, Zhimin Chen, Kris Ganjam, Yeye He, and W Redmond. 2016. Data services leveraging Bing’s data assets. IEEE Data Eng. Bull. 39, 3 (2016), 15–28.
- [11] Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. 2019. Tabfact: A large-scale dataset for table-based fact verification. arXiv preprint arXiv:1909.02164 (2019).
- [12] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311 (2022).
- [13] Xu Chu, Yeye He, Kaushik Chakrabarti, and Kris Ganjam. 2015. Tegra: Table extraction by global record alignment. In Proceedings of the 2015 ACM SIGMOD international conference on management of data. 1713–1728.

- [34] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023. Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing. Comput. Surveys 55, 9 (2023), 1–35.
- [35] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692

(2019).

- [36] Jayant Madhavan, Philip A Bernstein, and Erhard Rahm. 2001. Generic schema matching with cupid. In vldb, Vol. 1. 49–58.
- [37] Chris Mayfield, Jennifer Neville, and Sunil Prabhakar. 2010. ERACER: a database approach for statistical inference and data cleaning. In Proceedings of the 2010 ACM SIGMOD International Conference on Management of data. 75–86.
- [38] Sidharth Mudgal, Han Li, Theodoros Rekatsinas, AnHai Doan, Youngchoon Park, Ganesh Krishnan, Rohit Deep, Esteban Arcaute, and Vijay Raghavendra. 2018. Deep learning for entity matching: A design space exploration. In Proceedings of the 2018 International Conference on Management of Data. 19–34.
- [39] Avanika Narayan, Ines Chami, Laurel Orr, Simran Arora, and Christopher Ré.

2022. Can foundation models wrangle your data? arXiv preprint arXiv:2205.09911

(2022).

- [40] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems 35 (2022), 27730–27744.
- [41] George Papadakis, Ekaterini Ioannou, Emanouil Thanos, and Themis Palpanas.

2021. The four generations of entity resolution. Springer.

- [42] Panupong Pasupat and Percy Liang. 2015. Compositional semantic parsing on semi-structured tables. arXiv preprint arXiv:1508.00305 (2015).
- [43] Ralph Peeters and Christian Bizer. 2023. Using ChatGPT for Entity Matching. arXiv preprint arXiv:2305.03423 (2023).
- [44] Erhard Rahm and Philip A Bernstein. 2001. A survey of approaches to automatic schema matching. the VLDB Journal 10 (2001), 334–350.
- [45] Erhard Rahm, Hong Hai Do, et al. 2000. Data cleaning: Problems and current approaches. IEEE Data Eng. Bull. 23, 4 (2000), 3–13.
- [46] Anna Rogers, Olga Kovaleva, and Anna Rumshisky. 2021. A primer in BERTology: What we know about how BERT works. Transactions of the Association for Computational Linguistics 8 (2021), 842–866.
- [47] Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al.

2021. Multitask prompted training enables zero-shot task generalization. arXiv preprint arXiv:2110.08207 (2021).

- [48] Yoshihiko Suhara, Jinfeng Li, Yuliang Li, Dan Zhang, Çağatay Demiralp, Chen Chen, and Wang-Chiew Tan. 2022. Annotating columns with pre-trained language models. In Proceedings of the 2022 International Conference on Management of Data. 1493–1503.
- [49] Huan Sun, Hao Ma, Xiaodong He, Wen-tau Yih, Yu Su, and Xifeng Yan. 2016. Table cell search for question answering. In Proceedings of the 25th International Conference on World Wide Web. 771–782.
- [50] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023).
- [51] Jianhong Tu, Ju Fan, Nan Tang, Peng Wang, Guoliang Li, Xiaoyong Du, Xiaofeng Jia, and Song Gao. 2023. Unicorn: A unified multi-tasking model for supporting matching tasks in data integration. Proceedings of the ACM on Management of Data 1, 1 (2023), 1–26.
- [52] Gerrit JJ van den Burg, Alfredo Nazábal, and Charles Sutton. 2019. Wrangling messy CSV files by detecting row and type patterns. Data Mining and Knowledge Discovery 33, 6 (2019), 1799–1820.
- [53] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).
- [54] Gerardo Vitagliano, Mazhar Hameed, Lan Jiang, Lucas Reisener, Eugene Wu, and Felix Naumann. 2023. Pollock: A Data Loading Benchmark. Proceedings of the VLDB Endowment 16, 8 (2023), 1870–1882.
- [55] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171 (2022).
- [56] Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, David Wadden, Kelsey MacMillan, Noah A Smith, Iz Beltagy, et al. 2023. How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources. arXiv preprint arXiv:2306.04751 (2023).
- [57] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2022. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560 (2022).
- [58] Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. 2022. Super-naturalinstructions: Generalization

via declarative instructions on 1600+ nlp tasks. arXiv preprint arXiv:2204.07705

(2022).

- [59] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652 (2021).
- [60] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems 35

(2022), 24824–24837.

- [61] Jules White, Quchen Fu, Sam Hays, Michael Sandborn, Carlos Olea, Henry Gilbert, Ashraf Elnashar, Jesse Spencer-Smith, and Douglas C Schmidt. 2023. A prompt pattern catalog to enhance prompt engineering with chatgpt. arXiv preprint arXiv:2302.11382 (2023).
- [62] Xiaojun Xu, Chang Liu, and Dawn Song. 2017. Sqlnet: Generating structured queries from natural language without reinforcement learning. arXiv preprint arXiv:1711.04436 (2017).
- [63] Cong Yan and Yeye He. 2018. Synthesizing type-detection logic for rich semantic data types using open-source code. In Proceedings of the 2018 International Conference on Management of Data. 35–50.
- [64] Pengcheng Yin, Graham Neubig, Wen-tau Yih, and Sebastian Riedel. 2020. TaBERT: Pretraining for joint understanding of textual and tabular data. arXiv preprint arXiv:2005.08314 (2020).
- [65] Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev. 2018. Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Brussels, Belgium.
- [66] Chen Zhao and Yeye He. 2019. Auto-em: End-to-end fuzzy entity-matching using pre-trained deep models and transfer learning. In The World Wide Web Conference. 2413–2424.
- [67] Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. 2023. Lima: Less is more for alignment. arXiv preprint arXiv:2305.11206 (2023).

### A TASK DETAILS

#### Missing Value Identification (Column No Sep) Zero-Shot

Prompt: # Task Description: Please check the following table, there is one and exactly one cell in the table that is

missing. When you find this missing cell, please point it out using its column name. Return the final result as JSON in the format {"missing_col": "<missing column name>"}.

## Input: |Project|Team|Req| |---|---|---| |A|5|3|

- |I|3|2| |U|2|3| |2|1| |I|2|2|

Return the final result as JSON in the format {"missing_col": "<missing column name>"}. ## Output:

Completion: {"missing_col": "Project"}

#### Missing Value Identification (Column No Sep) Few-Shot

Prompt: # Task Description: Please check the following table, there is one and exactly one cell in the table that is

missing. When you find this missing cell, please point it out using its column name. Return the final result as JSON in the format {"missing_col": "<missing column name>"}.

## Input: |Project|Team|Req| |---|---|---|

- |A|4|1|

- |I|2|1| |O|3|3| |A|1| |E|4|2|

## Output: {"missing_col": "Req"}

## Input: |Project|Team|Req| |---|---|---|

- |I|2|1| |E|1|3| |A|1|3|

- |1|2| |E|4|2|

## Output: {"missing_col": "Project"}

## Input: |Project|Team|Req| |---|---|---|

|E|4|2|

- |O|2| |U|5|2|

- |I|4|2|

- |A|4|2|

## Output: {"missing_col": "Team"}

## Input: |Project|Team|Req| |---|---|---| |A|5|3|

- |I|3|2| |U|2|3|

- |2|1|

- |I|2|2|

Return the final result as JSON in the format {"missing_col": "<missing column name>"}. ## Output:

Completion: {"missing_col": "Project"}

#### Missing Value Identification (Row Sep) Zero-Shot

Prompt: # Task Description: Please check the following table, there is one and exactly one cell in the table that is

missing. When you find this missing cell, please point it out using the row id shown in the first column. Return the final result as JSON in the format {"row_id": "<row_id of the row with missing cell>"}.

## Input: |row_id|Project|Team|Req| |---|---|---|---|

- |1|A|5|3|

- |2|I|3|2|

- |3|U|2|3|

- |4||2|1|

- |5|I|2|2|

Return the final result as JSON in the format {"row_id": "<row_id of the row with missing cell>"}. ## Output:

Completion: {"row_id": "4"}

#### Missing Value Identification (Row Sep) Few-Shot

Prompt: # Task Description: Please check the following table, there is one and exactly one cell in the table that is

missing. When you find this missing cell, please point it out using the row id shown in the first column. Return the final result as JSON in the format {"row_id": "<row_id of the row with missing cell>"}.

## Input: |row_id|Project|Team|Req| |---|---|---|---|

- |1|A|4|1|

- |2|I|2|1|

- |3|O|3|3|

- |4|A|1||

- |5|E|4|2|

## Output: {"row_id": "4"}

## Input: |row_id|Project|Team|Req| |---|---|---|---|

- |1|I|2|1|

- |2|E|1|3|

- |3|A|1|3|

- |4||1|2|

- |5|E|4|2|

## Output: {"row_id": "4"}

## Input: |row_id|Project|Team|Req| |---|---|---|---|

- |1|E|4|2|

- |2|O||2|

- |3|U|5|2|

- |4|I|4|2|

- |5|A|4|2|

## Output: {"row_id": "2"}

## Input: |row_id|Project|Team|Req| |---|---|---|---|

- |1|A|5|3|

- |2|I|3|2|

- |3|U|2|3|

- |4||2|1|

- |5|I|2|2|

Return the final result as JSON in the format {"row_id": "<row_id of the row with missing cell>"}. ## Output:

Completion: {"row_id": "4"}

#### Missing Value Identification (Row No Sep) Zero-Shot

Prompt: # Task Description: Please check the following table, there is one and exactly one cell in the table that is

missing. When you find this missing cell, please point it out using the row id shown in the first column. Return the final result as JSON in the format {"row_id": "<row_id of the row with missing cell>"}.

## Input: |row_id|Project|Team|Req| |---|---|---|---|

- |1|A|5|3|

- |2|I|3|2|

- |3|U|2|3|

- |4|2|1|

- |5|I|2|2|

Return the final result as JSON in the format {"row_id": "<row_id of the row with missing cell>"}. ## Output:

Completion: {"row_id": "4"}

#### Missing Value Identification (Row No Sep) Few-Shot

Prompt: # Task Description: Please check the following table, there is one and exactly one cell in the table that is

missing. When you find this missing cell, please point it out using the row id shown in the first column. Return the final result as JSON in the format {"row_id": "<row_id of the row with missing cell>"}.

## Input: |row_id|Project|Team|Req| |---|---|---|---|

- |1|A|4|1|

- |2|I|2|1|

- |3|O|3|3|

- |4|A|1|

- |5|E|4|2|

## Output: {"row_id": "4"}

## Input: |row_id|Project|Team|Req| |---|---|---|---| |1|I|2|1| |2|E|1|3| |3|A|1|3| |4|1|2| |5|E|4|2|

## Output: {"row_id": "4"}

## Input: |row_id|Project|Team|Req| |---|---|---|---|

- |1|E|4|2|

- |2|O|2|

- |3|U|5|2|

- |4|I|4|2|

- |5|A|4|2|

## Output: {"row_id": "2"}

## Input: |row_id|Project|Team|Req|

|---|---|---|---|

- |1|A|5|3|

- |2|I|3|2|

- |3|U|2|3|

- |4|2|1|

- |5|I|2|2|

Return the final result as JSON in the format {"row_id": "<row_id of the row with missing cell>"}. ## Output:

Completion: {"row_id": "4"}

#### Missing Value Identification (Column Sep) Zero-Shot

Prompt: # Task Description: Please check the following table, there is one and exactly one cell in the table that is

missing. When you find this missing cell, please point it out using its column name. Return the final result as JSON in the format {"missing_col": "<missing column name>"}.

## Input: |Project|Team|Req| |---|---|---| |A|5|3|

- |I|3|2| |U|2|3| ||2|1| |I|2|2|

Return the final result as JSON in the format {"missing_col": "<missing column name>"}. ## Output:

Completion: {"missing_col": "Project"}

#### Missing Value Identification (Column Sep) Few-Shot

Prompt: # Task Description: Please check the following table, there is one and exactly one cell in the table that is

missing. When you find this missing cell, please point it out using its column name. Return the final result as JSON in the format {"missing_col": "<missing column name>"}.

## Input: |Project|Team|Req| |---|---|---|

- |A|4|1|

- |I|2|1|

- |O|3|3| |A|1|| |E|4|2|

## Output: {"missing_col": "Req"}

## Input: |Project|Team|Req| |---|---|---|

- |I|2|1| |E|1|3| |A|1|3|

- ||1|2| |E|4|2| ## Output: {"missing_col": "Project"} ## Input: |Project|Team|Req| |---|---|---| |E|4|2| |O||2| |U|5|2|

|I|4|2| |A|4|2|

## Output: {"missing_col": "Team"}

## Input: |Project|Team|Req| |---|---|---| |A|5|3| |I|3|2| |U|2|3|

- ||2|1|

- |I|2|2|

Return the final result as JSON in the format {"missing_col": "<missing column name>"}. ## Output:

Completion: {"missing_col": "Project"}

#### Column Finding Zero-Shot

Prompt: # Task Description: Please look at the table below and find the column that contains the given cell value.

Return the final result as JSON in the format {"result": "<name of the column containing the given cell value>"}.

## Input:

**Input Table:**

- |1|12|13|14|15|16|17|18| |---|---|---|---|---|---|---|---|

- |2|2|2|2|2|2|2|2|

- |3|3|3|3|3|3|3|3|

- |4|4|4|4|4|4|4|4|

- |5|5|5|5|5|5|5|5|

- |6|6|6|6|6|6|6|6|

- |7|7|7|7|7|7|7|7|

- |8|8|8|8|8|8|8|8|

- |9|9|9|9|9|9|9|9|

- |10|10|10|10|10|10|10|10|

- |11|11|11|11|11|11|11|11|

**Given Cell Value:** 2

Return the final result as JSON in the format {"result": "<name of the column containing the given cell value>"}. ## Output: Completion: {"result": "15"}

#### Column Finding Few-Shot

Prompt: # Task Description: Please look at the table below and find the column that contains the given cell value.

Return the final result as JSON in the format {"result": "<name of the column containing the given cell value>"}.

## Input:

**Input Table:** |price|crime|nox|rooms|dist|radial|proptax|stratio| |---|---|---|---|---|---|---|---| |17794|898299980163574|769999980926514|621000003814697|211999988555908|24|665999984741211|202000007629395| |21700|384999990463257|769999980926514|63899998664856|250999999046326|24|665999984741211|202000007629395| |22700|52020001411438|769999980926514|613000011444092|272000002861023|24|665999984741211|202000007629395| |22600|426100015640259|769999980926514|61100001335144|250999999046326|24|665999984741211|202000007629395| |24999|454199981689453|769999980926514|640000009536743|251999998092651|24|665999984741211|202000007629395| |19900|383699989318848|769999980926514|625|229999995231628|24|665999984741211|202000007629395| |20800|367799997329712|769999980926514|53600001335144|209999990463257|24|665999984741211|202000007629395| |16800|422200012207031|769999980926514|580000019073486|189999997615814|24|665999984741211|202000007629395| |21900|347399997711182|717999982833862|877999973297119|189999997615814|24|665999984741211|202000007629395| |27499|455600023269653|717999982833862|355999994277954|161000001430511|24|665999984741211|202000007629395|

**Given Cell Value:** 426100015640259

## Output: {"result": "crime"}

## Input:

**Input Table:** |Player|Class|Team|GP|G|A|Pts|PIM| |---|---|---|---|---|---|---|---| |Nathan Gerbe|Junior|Boston College|43|35|33|68|65| |Kevin Porter|Senior|Michigan|43|33|30|63|18| |Chad Kolarik|Senior|Michigan|39|30|26|56|24| |Ryan Lasch|Sophomore|St. Cloud State|40|25|28|53|12| |Simon Lambert|Senior|RIT|37|21|30|51|40| |Joe Whitney|Freshman|Boston College|44|11|40|51|50| |Ben Smith|Sophomore|Boston College|44|25|25|50|12| |Ryan Jones|Senior|Miami (OH)|42|31|18|49|83| |Ryan Cruthers|Senior|Robert Morris|34|22|27|49|40| |Matt Fornataro|Senior|New Hampshire|38|18|28|46|52|

**Given Cell Value:** Miami (OH)

## Output: {"result": "Team"}

## Input:

**Input Table:** |Nation|1977|1995|1997|1999|2001|2003|2005|2008|2010|2012|2014|1979|2016|Years|1981|1983|1985|1987|1989| |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---| |Algeria|nan|nan|nan|nan|nan|19th|nan|16th|nan|nan|nan|nan|nan|4|nan|nan|nan|nan|12th| |Angola|nan|13th|17th|17th|15th|nan|15th|11th|14th|19th|21st|nan|14th|10|nan|nan|nan|nan|nan| |Argentina|nan|17th|nan|nan|16th|nan|20th|12th|15th|20th|20th|nan|16th|8|nan|nan|nan|nan|nan| |Australia|nan|nan|nan|nan|nan|nan|nan|20th|22nd|nan|nan|nan|nan|2|nan|nan|nan|nan|nan| |Austria|14th|nan|nan|nan|nan|nan|nan|nan|nan|11th|nan|11th|19th|8|nan|nan|14th|nan|14th| |Belarus|nan|nan|nan|nan|nan|nan|nan|nan|nan|nan|nan|nan|nan|1|nan|nan|nan|nan|nan| |Brazil|nan|13th|12th|12th|13th|15th|9th|9th|12th|12th|15th|nan|11th|13|nan|nan|nan|nan|nan| |Bulgaria|nan|9th|nan|nan|nan|nan|nan|nan|nan|nan|nan|nan|nan|5|nan|5th|nan|nan|3rd| |Canada|nan|nan|nan|20th|nan|nan|nan|nan|nan|nan|nan|nan|nan|3|8th|nan|nan|nan|nan| |Chile|nan|nan|nan|nan|nan|nan|nan|nan|nan|nan|nan|nan|22nd|1|nan|nan|nan|nan|nan|

**Given Cell Value:** 13

## Output: {"result": "Years"}

## Input:

**Input Table:**

- |1|12|13|14|15|16|17|18| |---|---|---|---|---|---|---|---|

- |2|2|2|2|2|2|2|2|

- |3|3|3|3|3|3|3|3|

- |4|4|4|4|4|4|4|4|

- |5|5|5|5|5|5|5|5|

- |6|6|6|6|6|6|6|6|

- |7|7|7|7|7|7|7|7|

- |8|8|8|8|8|8|8|8|

- |9|9|9|9|9|9|9|9|

- |10|10|10|10|10|10|10|10|

- |11|11|11|11|11|11|11|11|

**Given Cell Value:** 2

Return the final result as JSON in the format {"result": "<name of the column containing the given cell value>"}. ## Output: Completion: {"result": "15"}

#### Table-QA Zero-Shot

Prompt: # Task Description: Please look at the table, and then answer the question. Please also provide an explanation

on your answer. Return the final result as JSON in the format {"answer": "<YOUR ANSWER>"}. ## Input:

*Table*

|Rank|Nation|Gold|Silver|Bronze|Total| |---|---|---|---|---|---|

- |1|Netherlands|8|3|1|12|

- |2|Australia|3|3|4|10|

- |3|United States|2|5|1|8|

- |4|Hungary|1|1|3|5|

- |5|Canada|1|-|3|4|

- |6|Italy|-|2|1|3|

- |7|Russia|-|1|1|2|

- |8|China|-|-|1|1|

*Question:* how many nations are there?

Return the final result as JSON in the format {"answer": "<YOUR ANSWER>"}. ## Output:

Completion: {"answer": "8"}

#### Table-QA Few-Shot

Prompt: # Task Description: Please look at the table, and then answer the question. Please also provide an explanation

on your answer. Return the final result as JSON in the format {"answer": "<YOUR ANSWER>"}. ## Input:

*Table* |Rank|Heat|Nationality|Time|Notes| |---|---|---|---|---|

- |1|1|Russia|23.11|Q|

- |2|1|Belgium|23.23|Q|

- |3|2|Russia|23.37|Q|

- |4|1|Poland|23.39|Q, SB|

- |5|1|Ukraine|23.4|Q, SB|

- |6|2|United States Virgin Islands|23.49|Q|

- |7|1|Belarus|23.52|nan|

- |8|1|Canada|23.62|nan|

- |9|2|Poland|23.69|Q|

- |10|2|Hungary|23.87|Q|

- |11|2|Ireland|23.89|nan|

- |12|1|Mexico|23.96|nan|

- |13|2|Lithuania|24.09|nan|

- |14|2|Brazil|24.18|nan|

- |15|2|Great Britain|24.31|nan|

- |16|1|Italy|24.4|nan|

- *Question:* who is the only cyclist from brazil? ## Output: {"answer": "Raquel da Costa"} ## Input:

- *Table* |Eps #|Prod #|Title|Air Date| |---|---|---|---|

- |1|2|Menace Of The Mole Men|9/9/1967|

- |2|3|Diablo|9/16/1967|

- |3|7|The Way It All Began|9/23/1967|

- |4|5|Invasion Of The Super-Skrull|9/30/1967|

- |5a|1|Klaws|10/7/1967|

- |5b|4|The Red Ghost|10/7/1967|

- |6|9|Prisoners Of Planet X|10/14/1967|

- |7|14|It Started On Yancy Street|10/21/1967|

- |8|6|Three Predictions Of Dr. Doom|10/28/1967|

- |9|8|Behold A Distant Star|11/4/1967|

- |10|12|Demon in the Deep|11/11/1967|

- |11|11|Danger In The Depths|11/18/1967|

- |12|13|Return Of the Mole Man|11/25/1967|

- |13|19|Rama-Tut|12/9/1967|

- |14|15|Galactus|12/16/1967|

- |15|16|The Micro World Of Dr. Doom|12/30/1967|

- |16|17|Blastaar, The Living Bomb-Burst|1/6/1968|

- |17|10|The Mysterious Molecule Man|1/13/1968|

- |18|18|The Terrible Tribunal|9/14/1968|

- |19|20|The Deadly Director|9/21/1968|

*Question:* how long did the show air?

Return the final result as JSON in the format {"answer": "<YOUR ANSWER>"}. ## Output:

Completion: {"answer": "1 year"}

#### Column Type Annotation Zero-Shot

Prompt: # Task Description: Please look at the input column and determine the semantic type that can describe *every

single* instance the input column. Please only choose one semantic type from the candidate list, and remember that the type you choose has to accurately describe every single entity in the column. If no candidate column type can suitably describe every single instance in the column, please return 'None'. Please only choose one type from the candidate list below, and *do not* create new types. Return the final result as JSON in the format {"chosen_semantic_type": "<an entry from the candidate list or None>"}.

## Input:

**Column:** |Party| |---| |Liberal| |Conservative|

**Candidate column type:** AcademicJournal BaseballPlayer Book City Company Continent Film Mayor Monarch Mountain Newspaper PoliticalParty

Scientist SportsTeam

Return the final result as JSON in the format {"chosen_semantic_type": "<an entry from the candidate list or

None>"}. ## Output: Completion: {"chosen_semantic_type": "PoliticalParty"}

#### Column Type Annotation Few-Shot

Prompt: # Task Description: Please look at the input column and determine the semantic type that can describe *every

single* instance the input column. Please only choose one semantic type from the candidate list, and remember that the type you choose has to accurately describe every single entity in the column. If no candidate column type can suitably describe every single instance in the column, please return 'None'. Please only choose one type from the candidate list below, and *do not* create new types. Return the final result as JSON in the format {"chosen_semantic_type": "<an entry from the candidate list or None>"}.

## Input:

**Column:** |Name| |---| |Wells Fargo Tower| |Regions-Harbert Plaza| |AT&T City Center| |Regions Center| |City Federal Building| |Alabama Power Headquarters Building| |Thomas Jefferson Tower| |John Hand Building| |Daniel Building|

**Candidate column type:** AcademicJournal Airport Book Building City Film Mammal Newspaper Plant PoliticalParty Scientist SportsTeam ## Output: {"chosen_semantic_type": "Building"} ## Input:

**Column:** |Team| |---| |Minnesota 1| |New York|

**Candidate column type:** AdministrativeRegion Continent Mayor Saint University Wrestler Writer

## Output: {"chosen_semantic_type": "AdministrativeRegion"}

## Input:

**Column:** |Engine| |---| |Honda| |Honda| |Honda| |Chevrolet| |Chevrolet|

**Candidate column type:** AcademicJournal Company Currency Film Lake Saint Writer

## Output: {"chosen_semantic_type": "Company"}

## Input:

**Column:** |Name| |---| |Juliomys| |Pipanacoctomys| |Salinoctomys| |Tapecomys| |Hyladelphys| |Handleyomys| |Sommeromys| |Chacodelphys| |Drymoreomys|

**Candidate column type:** Building City Company Continent Country Currency

## Output: {"chosen_semantic_type": "None"}

## Input:

**Column:** |Service| |---| |Star FM| |Radio Teddy| |Berliner Rundfunk 91|

- |4| |Jam FM| |94| |3 rs2| |Radio Eins| |Deutschlandfunk| |98.8 KISS FM Berlin| |Antenne Brandenburg| |FluxFM| |Klassik Radio| |Radyo Metropol FM| |Fritz| |Energy Berlin| |104.6 RTL| |105'5 Spreeradio| |Radio B2|

**Candidate column type:** BaseballPlayer Building Lake Mammal Mayor PoliticalParty SportsTeam ## Output: {"chosen_semantic_type": "None"} ## Input:

**Column:** |Common name| |---| |White-collared swift|

**Candidate column type:** Airport Book Language Monarch Plant PoliticalParty University ## Output: {"chosen_semantic_type": "None"}

## Input:

**Column:** |Party| |---| |Liberal| |Conservative|

**Candidate column type:** AcademicJournal BaseballPlayer Book City Company Continent Film Mayor Monarch Mountain Newspaper PoliticalParty Scientist SportsTeam

Return the final result as JSON in the format {"chosen_semantic_type": "<an entry from the candidate list or

None>"}. ## Output: Completion: {"chosen_semantic_type": "PoliticalParty"}

#### Error Detection Zero-Shot

Prompt: Description: Please examine the input table and let me know which cell or cells are erroneous. If there are

multiple erroneous cells, return them all in a list. If there is no erroneous cell, return 'None'. Please only return the erroneous cell or cells with high confidence. Return the final result as JSON in the format {"erroneous_cells": "<a list of erroneous cells or None>"}. Let's think step by step and show your reasoning

before showing the final result.

Question: |Province|Teams|Rank| |---|---|---| |Bengkulu|1|9| |Lampung|1|9| |Banetn|1|9| |Central MJava|1|9| |Yogyakarta|1|9|

Let's think step by step and show your reasoning before showing the final result. Return the final result as

JSON in the format {"erroneous_cells": "<a list of erroneous cells or None>"}. Answer: Completion: The cell "Banetn" is misspelled and should be "Banten". The cell "Central MJava" is misspelled and should be "

Central Java". Therefore, the answer is {"erroneous_cells": ["Banetn", "Central MJava"]}

#### Error Detection Few-Shot

Prompt: # Task Description: Kindly evaluate the input table and let me know about any cell or cells that have errors. If

there are multiple cells with errors, provide them in a list. If no cells have errors, indicate 'None'. Only return the cells for which you have a high level of confidence in their error status. Return the final result as JSON in the format {"erroneous_cells": "<a list of erroneous cells or None>"}. Let's think step by

step and show your reasoning before showing the final result.

In: |District|Party|First Election| |---|---|---|

- |12|Repp|2011| |43|Dem|2003| |24|Rep|2002|

Out: The cell "Repp" is misspelled and should be "Rep". Therefore, the answer is {"erroneous_cells": ["Repp"]}

In: |District|Party|First Election| |---|---|---|

- |1|Rep|1993|

- |2|Rep|2015|

- |3|Rep|2009|

- |4|Rep|2013|

- |5|Rep|2011|

Let's think step by step and show your reasoning before showing the final result. Return the final result as JSON in the format {"erroneous_cells": "<a list of erroneous cells or None>"}.

Out: Completion: All cells in the table seem to be correct and no apparent error is found. Therefore, the final answer is {"

erroneous_cells": "None"}

#### Row/Column Swapping Zero-Shot

Prompt: # Task Description: Please swap the second row and the third row in the table. Please return the table after

swapping.

Q: |Value|Colour|Type| |---|---|---|

- |1 lepton|brown|a|

- |2 lepta|grey|a|

- |3 lepta|orange|a| |5 lepta|green|b| |10 lepta|red|b| A:

Completion: |Value|Colour|Type| |---|---|---|

- |1 lepton|brown|a|

- |3 lepta|orange|a|
- |2 lepta|grey|a| |5 lepta|green|b| |10 lepta|red|b|

#### Row/Column Swapping Few-Shot

Prompt: Objective: If possible, could you swap the positions of column "Group" and column "%" in the table? Let me know

the outcome.

## Input: |Group|N|%| |---|---|---| |IIG|0|0.0%| |IIIAB|11|1.1%| |IVB|0|0.0%| |IIF|1|0.1%| |IIE|2|0.2%|

## Output: |%|N|Group| |---|---|---|

- |0.0%|0|IIG|

- |1.1%|11|IIIAB|

- |0.0%|0|IVB|

- |0.1%|1|IIF|

- |0.2%|2|IIE|

## Input: |Group|N|%| |---|---|---| |IAB complex|10|1.0%|

- |IC|0|0.0%| |IIAB|6|0.6%|

- |IIC|0|0.0%| |IID|3|0.3%| ## Output:

Completion: |%|N|Group| |---|---|---| |1.0%|10|IAB complex|

- |0.0%|0|IC|

- |0.6%|6|IIAB|

- |0.0%|0|IIC|

- |0.3%|3|IID|

Schema Matching Zero-Shot Prompt:

Description: Please identify the matching columns between Table A and Table B. For each column in Table A, specify the corresponding column in Table B. If a column in A has no corresponding column in Table B, you can map it to None. Represent each column mapping using a pair of column headers in a list, i.e., [Table A Column, Table B column or None]. Provide the mapping for each column in Table A and return all mappings in a

list. Return the final result as JSON in the format {"column_mappings": "<a list of column pairs>"}. Question:

- Table A: |IATA|ICAO|Country| |---|---|---| |YYC|CYYC|Canada| |BNE|YBBN|Australia|

- Table B: |Nation|Airport Code| |---|---| |Thailand|DMK| |Philippines|CEB| |Philippines|CRK|

Return the final result as JSON in the format {"column_mappings": "<a list of column pairs>"}. Answer:

Completion: {"column_mappings": [["IATA", "Airport Code"], ["ICAO", "None"], ["Country", "Nation"]]}

#### Schema Matching Few-Shot

Prompt: Description: Find the corresponding columns between Table A and Table B. Specify the related columns in Table B

for each column present in Table A. If a column in Table A has no corresponding column in Table B, you can denote it as None. Use pairs of column headers in a list structure, like [Table A Column, Table B Column or None], to depict each mapping. Present the mappings for all the columns in Table A and organize them into a list Return the final result as JSON in the format {"column_mappings": "<a list of column pairs>"}.

Question:

- Table A: |Dignity|Detriment| |---|---| |Gemini and Virgo|Sagittarius and Pisces| |Leo|Aquarius|

- Table B: |Celestial Body (Symbol)|High Point| |---|---| |Moon ()|Taurus| |Venus ()|Pisces| |Mars ()|Capricorn|

Answer: {"column_mappings": [["Dignity", "None"], ["Detriment", "None"]]}

Question:

- Table A: |Barony|Civil parish|Acres|Poor law union|County| |---|---|---|---|---| |Callan|Callan|21|Callan|Kilkenny|

- |Callan|Callan|xx|Callan|Kilkenny|
- Table B: |Location|Parish|Surface Area|Subdivision| |---|---|---|---| |Caherlesk|Ballytobin|1,162|Kells| |CLOGH T.|Castlecomer|xx|Fassadinin| |CASTLECOMER T.|Castlecomer|xx|Fassadinin|

Return the final result as JSON in the format {"column_mappings": "<a list of column pairs>"}. Answer:

Completion: {"column_mappings": [["Barony", "Subdivision"], ["Civil parish", "Parish"], ["Acres", "Surface Area"], ["Poor

law union", "None"], ["County", "None"]]}

#### Header Value Matching Zero-Shot

Prompt: Task: Given the input table data and the list of candidate headers, please determine the most suitable column

header for each column in the table. Please only choose column headers from the candidate list. Please only return the most suitable column header for each column. Return the chosen column headers in a list. Do not return the entire table. Return the final result as JSON in the format {"column_headers": "<a list of headers for each column chosen from the candidate list>"}.

[Q]:

**Table Data:** |||| |---|---|---|

- |1|681|Multihull| |3|911|Keelboat|

- |2|947|Dinghy| |nan|920|Dinghy|

- |1|870|Dinghy|

**Candidate column headers:**

- - Portsmouth Number

- - crew

- - Type

Return the final result as JSON in the format {"column_headers": "<a list of headers for each column chosen from the candidate list>"}.

[A]: Completion: {"column_headers": ["crew", "Portsmouth Number", "Type"]}

#### Header Value Matching Few-Shot

Prompt: Objective: With the input table data and the list of potential headers at hand, make determinations about the

most fitting column header for every table column. Only select column headers from the candidate list and compile them into a list. Return the final result as JSON in the format {"column_headers": "<a list of headers for each column chosen from the candidate list>"}.

## Input:

**Table Data:**

|||| |---|---|---|

- |Cust 1|126|1|

- |Cust 2|65|2| |Cust 5|30|5|

- |Cust 3|45|3| |Cust 9|19|9|

**Candidate column headers:**

- - Count

- - Custom

- - Name

## Output: {"column_headers": ["Name", "Count", "Custom"]}

## Input:

**Table Data:** |||| |---|---|---|

- |1|The Beginning|1:48|

- |15|Ride Slow|3:05| |7|Siri (Skit)|0:53|

- |5|Wrong n Tough|2:45|

- |9|Officer Hatchet|3:46|

**Candidate column headers:**

- - Name

- - Length

- - Track #

Return the final result as JSON in the format {"column_headers": "<a list of headers for each column chosen from

the candidate list>"}. ## Output: Completion: {"column_headers": ["Track #", "Name", "Length"]}

#### Column Augmentation Zero-Shot

Prompt: Task: Please generate a new additional column for the input table and append the new column to the right of the

table. Return the new table with the additional column.

Input: |Number|Name| |---|---|

- |279|Hazel|

- |280|Audrey|

- |281|Gwen|

- |282|Doris|

- |283|Mona|

- |284|Vera|

- |285|Car No 85|

- |286|Car No 86|

- |287|Car No 87|

- |288|Car No 88|

Output: Completion:

|Number|Name|Note| |---|---|---|

- |279|Hazel|5-BEL Trust, under restoration|

- |280|Audrey|VSOE, in service|

- |281|Gwen|VSOE, in service|

- |282|Doris|5-BEL Trust, to be restored|

- |283|Mona|VSOE, stored|

- |284|Vera|VSOE, in service|

- |285|Car No 85|5BEL Trust, under restoration|

- |286|Car No 86|VSOE, stored|

- |287|Car No 87|5-BEL Trust, under restoration|

- |288|Car No 88|5-BEL Trust, under restoration|

#### Column Augmentation Few-Shot

Prompt: Task: Generate a new column for the input table and attach it on the right side. Share the updated table with

the appended column.

## Input: |Year|Peak positions(CAN Country)| |---|---| |1981|41| |1983|40|

- |1983|35|

- |1984|39|

- |1985|29|

- |1986|10|

- |1987|20|

- |1988|9|

- |1989|30| |1989|9|

## Output: |Year|Peak positions(CAN Country)|Album| |---|---|---| |1981|41|Non-album songs| |1983|40|Non-album songs|

- |1983|35|Non-album songs|

- |1984|39|Touch My Heart|

- |1985|29|Touch My Heart|

- |1986|10|Touch My Heart|

- |1987|20|Touch My Heart|

- |1988|9|Touch My Heart|

- |1989|30|Touch My Heart| |1989|9|Touch My Heart|

## Input: |end km|home km| |---|---|

- |0.8|1.402|

- |1.402|1.44|

- |1.44|1.884| |1.884|6.071| |6.071|6.513| |6.513|8.3| |8.3|12.4| |12.4|14.6| |14.6|21.0| |21.0|168.7| ## Output:

Completion: |end km|home km|km/h| |---|---|---|

- |0.8|1.402|110|

- |1.402|1.44|35| |1.44|1.884|65| |1.884|6.071|110| |6.071|6.513|105| |6.513|8.3|110| |8.3|12.4|165| |12.4|14.6|200| |14.6|21.0|270| |21.0|168.7|300|

#### Natural-Language to SQL Zero-Shot

Prompt: Description: Please write the SQL query given the input table and question. Use 'table' as the table name if

needed. Return the final result as JSON in the format {"SQL": "<SQL code>"}. In:

**Input table:** |Rank|Group|Name|Nationality|Mark| |---|---|---|---|---|

- |1|B|Brittney Reese|United States|6.87|

- |2|A|Maurren Higa Maggi|Brazil|6.79|

- |3|B|Tatyana Lebedeva|Russia|6.70|

- |4|A|Carolina Kluft|Sweden|6.70|

- |5|A|Grace Upshaw|United States|6.68|

**Question:** What rank does Tricia Flores have?

Return the final result as JSON in the format {"SQL": "<SQL code>"}. Out:

Completion: {"SQL": "SELECT `Rank` FROM table WHERE `Name` = tricia flores"}

#### Natural-Language to SQL Few-Shot

Prompt: Description: Please write an SQL query using the input table and question. If needed, utilize 'table' as the

table name. Return the final result as JSON in the format {"SQL": "<SQL code>"}. Q:

**Input table:** |Date|Score|Set 1|Set 2|Set 3|Total| |---|---|---|---|---|---|

- |13 Nov|3-0|15-6|15-12|15-13|45-31|

- |13 Nov|0-3|6-15|13-15|6-15|25-45|

- |14 Nov|0-3|2-15|8-15|6-15|16-45|

- |14 Nov|3-2|15-11|10-15|16-14|67-64|

- |15 Nov|3-1|15-9|15-11|13-15|58-47|

**Question:** What is the score for set 2 when set 3 was 15-6?

A: {"SQL": "SELECT `Set 2` FROM table WHERE `Set 3` = 15-6"}

Q:

**Input table:** |Position|Team|Played|Wins|Draws|Losses|Scored|Conceded|Points| |---|---|---|---|---|---|---|---|---|

- |1|Olimpia|9|4|4|1|14|6|16|

- |2|12 de Octubre|9|3|6|0|14|8|15|

- |3|Cerro Porteno|9|3|5|1|16|10|14|

- |4|Guarani|9|3|5|1|8|6|14|

- |5|Cerro Cora|9|3|4|2|9|8|13|

**Question:** How many losses for the team atl. colegiales?

Return the final result as JSON in the format {"SQL": "<SQL code>"}. A:

Completion: {"SQL": "SELECT MAX `Losses` FROM table WHERE `Team` = Atl. Colegiales"}

#### Table Summary Zero-Shot

Prompt: Objective: Please look at the table below and provide a title for the table. Return the final result as JSON in

the format {"summary": "<summary of table>"}.

## Input: |Unnamed: 0|Day 8|Nominations received|Day 9|Day 10|Day 13| |---|---|---|---|---|---| |Jenny|Lucy|1.0|Georgina|Lucy|Natalia| |Natalia|In Secret Loft|13.0|Georgina|Lucy|Manuel| |Marijke|Jan|18.0|Jan|Lucy|Fancy| |Simon|Marijke|5.0|Marijke|Marijke|Marijke| |Martin|Lucy|3.0|Lucy|Lucy|Natalia|

Return the final result as JSON in the format {"summary": "<summary of table>"}. ## Output:

Completion: {"summary": "Promi Big Brother (season 1)"}

#### Table Summary Few-Shot

Prompt: Objective: Your task is to give a summary of the input table's main information. Return the final result as JSON

in the format {"summary": "<summary of table>"}.

In: |Survivor|Day 1|Day 2|Day 3|Day 4|Day 5|Day 6|Day 7|Missions| |---|---|---|---|---|---|---|---|---| |Amena|SAFE|SAFE|SAFE|SAFE|SAFE|SAFE|WON|7| |Leah|SAFE|SAFE|SAFE|SAFE|SAFE|SAFE|WON|6| |Megan|SAFE|SAFE|SAFE|SAFE|SAFE|SAFE|WON|7| |Nic|NOT IN GAME|NOT IN GAME|NOT IN GAME|SAFE|SAFE|SAFE|WON|5| |Aston|SAFE|SAFE|SAFE|SAFE|SAFE|SAFE|DIED|7|

Out: {"summary": "I Survived a Zombie Apocalypse"}

In: |Name|Charter/Plan|Type|Class|County| |---|---|---|---|---| |Bensalem|Plan|Township|Second|Bucks| |Bethlehem|Charter|City|Third|Lehigh Northampton| |Bristol|Plan|Township|First|Bucks| |College|Plan|Township|Second|Centre| |DuBois|Plan|City|Third|Clearfield|

Return the final result as JSON in the format {"summary": "<summary of table>"}. Out:

Completion: {"summary": "List of Pennsylvania municipalities and counties with home rule charters, optional charters, or

optional plans"}

#### List Extraction Zero-Shot

Prompt: Objective: Please transform the list below into a table with multiple columns. Please return the table using

plain text. Use vertical bars (|) to separate columns and use a new line for each row. Return the final result as JSON in the format {"table": "<table transformed from the list>"}.

[Q]: JobFamily OrgID minTiG maxTiG

- JF1 5.0 2.0 12.0 JF3 1.0 2.0 16.0

- JF2 4.0 2.0 7.0 JF1 9.0 2.0 7.0

- JF3 41.0 2.0 7.0

Return the final result as JSON in the format {"table": "<table transformed from the list>"}. [A]:

Completion: {"table": "|JobFamily|OrgID|minTiG|maxTiG|\n|---|---|---|---|\n|JF1|5.0|2.0|12.0|\n|JF3|1.0|2.0|16.0|\n|JF2

|4.0|2.0|7.0|\n|JF1|9.0|2.0|7.0|\n|JF3|41.0|2.0|7.0|\n"}

#### List Extraction Few-Shot Prompt:

Description: Change the list provided into a table with several columns. The table must be represented in plain text, with vertical bars (|) separating columns and a new line for every row. Return the final result as JSON in the format {"table": "<table transformed from the list>"}.

[Q]: Ch Video (MHz) Audio (MHz)

- 15 487.25 493.75 36 693.25 699.75 39 717.25 723.75 23 551.25 557.75 58 871.25 877.75

[A]: {"table": "|Ch|Video (MHz)|Audio (MHz)|\n|---|---|---|\n|15|487.25|493.75|\n|36|693.25|699.75|\n

|39|717.25|723.75|\n|23|551.25|557.75|\n|58|871.25|877.75|\n"}

[Q]: Month name amount 5 c 3

- 4 e 2

- 3 b 6

- 2 e 2

- 3 a 3

Return the final result as JSON in the format {"table": "<table transformed from the list>"}. [A]:

Completion: {"table": "|Month|name|amount|\n|---|---|---|\n|5|c|3|\n|4|e|2|\n|3|b|6|\n|2|e|2|\n|3|a|3|\n"}

#### Row Augmentation Zero-Shot

Prompt: Task: Please generate a new row for the input table and append it at the bottom of the table. Return the new

table with the additional generated row.

[Q]: |Rank|Player|Receptions| |---|---|---|

- |1|Jerry Rice|1,549|

- |2|Tony Gonzalez|1,325|

- |3|Larry Fitzgerald|1,125|

- |4|Marvin Harrison|1,102|

- |5|Cris Carter|1,101|

- |6|Tim Brown|1,094|

- |7|Jason Witten|1,089|

- |8|Terrell Owens|1,078|

- |9|Anquan Boldin|1,076| [A]:

Completion: |Rank|Player|Receptions| |---|---|---|

- |1|Jerry Rice|1,549|

- |2|Tony Gonzalez|1,325|

- |3|Larry Fitzgerald|1,125|

- |4|Marvin Harrison|1,102|

- |5|Cris Carter|1,101|

- |6|Tim Brown|1,094|

- |7|Jason Witten|1,089|

- |8|Terrell Owens|1,078|

- |9|Anquan Boldin|1,076|

- |10|Reggie Wayne|1,070|

#### Row Augmentation Few-Shot

Prompt: Description: Please generate a new row for the input table and append it at the bottom of the table. Return the

new table with the additional generated row.

[Q]: |ProductID|Month|TargetSale| |---|---|---|

- |1|6|50|

- |1|7|60|

- |1|8|70|

- |2|6|40|

|2|7|50| |2|8|60| |3|6|30| |3|7|40|

- |3|8|50|

[A]: |ProductID|Month|TargetSale| |---|---|---|

- |1|6|50|

- |1|7|60|

- |1|8|70| |2|6|40| |2|7|50| |2|8|60|

- |3|6|30|

- |3|7|40|

- |3|8|50|

- |4|6|30|

[Q]: |Rank|Player|2015 FF Points|Games|Avg| |---|---|---|---|---|

- |1|Cam Newton|389.1|16|24.3|

- |2|Tom Brady|343.7|16|21.5|

- |3|Russell Wilson|336.4|16|21.0|

- |4|Blake Bortles|316.1|16|19.8|

- |5|Carson Palmer|309.2|16|19.3|

- |7|Aaron Rodgers|301.3|16|18.8|

- |8|Kirk Cousins|293.5|16|18.3|

- |9|Matthew Stafford|289.7|16|18.1|

- |10|Eli Manning|287.6|16|18.0| [A]:

Completion: |Rank|Player|2015 FF Points|Games|Avg| |---|---|---|---|---|

- |1|Cam Newton|389.1|16|24.3|

- |2|Tom Brady|343.7|16|21.5|

- |3|Russell Wilson|336.4|16|21.0|

- |4|Blake Bortles|316.1|16|19.8|

- |5|Carson Palmer|309.2|16|19.3|

- |7|Aaron Rodgers|301.3|16|18.8|

- |8|Kirk Cousins|293.5|16|18.3|

- |9|Matthew Stafford|289.7|16|18.1|

- |10|Eli Manning|287.6|16|18.0|

- |11|Ryan Fitzpatrick|285.1|16|17.8|

Row/Column Filtering Zero-Shot

Prompt: # Task Description: Please select the first row in the table. Please return the table with the selected rows. Question: |sk_usuario|nk_id_usuario|nome_usuario| |---|---|---| |1|1|tatsurok2018| |4|4|Wilfredo M|

- |12|12|Jonathan C|

- |19|19|kenthecelt|

- |20|20|josenc2016| Answer:

Completion: |sk_usuario|nk_id_usuario|nome_usuario| |---|---|---| |1|1|tatsurok2018|

#### Row/Column Filtering Few-Shot

Prompt: Objective: Could you choose column "Option_content" in the table? After the selection, please return the table. In: |question_id|Ans_id|Option_content| |---|---|---|

- |94|2|False| |111|1|False| |118|2|False| |57|1|False|

- |95|1|False|

Out: |Option_content| |---| |False| |False| |False| |False| |False|

In: |question_id|Ans_id|Option_content| |---|---|---|

- |16|1|False|

- |17|1|False|

- |18|1|False|

- |19|2|False|

- |20|2|False| Out:

Completion: |Option_content| |---| |False| |False| |False| |False| |False|

#### Row-to-Row Transformation Zero-Shot

Prompt: Task: You are given a table of inputs and outputs in two columns. Please figure out the patterns between inputs

and outputs from the first few rows and then determine the output value for the last row (denoted by '[ Output Value]'). Please only return the output value. Return the final result as JSON in the format {" output_value": "<output value transformed from the input value>"}.

[Q]: |Input|Output| |---|---| |18:1|1| |14:6|6|

- |9:11|11|

- |10:12|12| |9:15|[Output Value]|

Return the final result as JSON in the format {"output_value": "<output value transformed from the input value >"}.

[A]: Completion: {"output_value": "15"}

#### Entity Matching Zero-Shot

Prompt: Instruction: Please determine whether the First Cosmetic and the Second Cosmetic refer to the same entity or not.

Your final answer should be 'Yes' or 'No'. Return the final result as JSON in the format {"answer": "<Yes or No>"}. Let's think step by step and show your reasoning before showing the final result.

Question: the First Cosmetic: |Price|Color|Description| |---|---|---|

|$16.50|Wheat |CLINIQUE brow shaper| the Second Cosmetic can be presented as follows. |Color|Description|Price| |---|---|---| |Charcoaled |Clinique Brow Shaper Charcoaled|$29.90| Let's think step by step and show your reasoning before showing the final result. Return the final result as

JSON in the format {"answer": "<Yes or No>"}. Answer: Completion: the First Cosmetic and the Second Cosmetic are different entities because they have different prices and colors.

the First Cosmetic is priced at $16.50 and is Wheat in color, while the Second Cosmetic is priced at $29.90 and is Charcoaled in color. Therefore, the final answer is {"answer": "No"}.

#### Entity Matching Few-Shot

Prompt: Instruction: Please verify whether Cosmetic I and Cosmetic II are denoting the same entity. Indicate your

conclusive answer as either 'Yes' or 'No'. Return the final result as JSON in the format {"answer": "<Yes or No>"}. Let's think step by step and show your reasoning before showing the final result.

In: Here is the description of Cosmetic I. |Description|Price|Color| |---|---|---| |Hourglass extreme sheen high shine lip gloss|$28.00|Child|

Cosmetic II: |Price|Color|Description| |---|---|---| |$41.35|Truth|Hourglass Extreme Sheen High Shine Lip Gloss Truth|

Out: Cosmetic I and Cosmetic II are different entities because they have different prices ($28.00 vs. $41.35) and

colors (Child vs. Truth). Therefore, the final answer is {"answer": "No"}.

In: Here is the description of Cosmetic I. |Description|Price|Color| |---|---|---| |tarte lipsurgence lip tint|$24.00|Grateful|

Cosmetic II: |Price|Color|Description| |---|---|---| |$35.40|Energy|Tarte Lipsurgence Skintuitive Lip Tint - Energy (.10 oz)|

Out: Cosmetic I and Cosmetic II are different entities because they have different prices, colors, and descriptions.

Cosmetic I is a Tarte Lipsurgence Lip Tint with a price of $24.00 and a color of Grateful. Cosmetic II is a Tarte Lipsurgence Skintuitive Lip Tint with a price of $35.40 and a color of Energy. Therefore, the final answer is {"answer": "No"}.

In: Here is the description of Cosmetic I.

|Description|Price|Color| |---|---|---| |Bobbi Brown foundation brush|$42.00|nan|

Cosmetic II: |Price|Color|Description| |---|---|---| |$69.00|nan|Bobbi Brown Foundation Brush|

Out: Cosmetic I and Cosmetic II refer to the same entity as they both have the same description, "Bobbi Brown

Foundation Brush", and the same color, nan. The only difference is the price, with Cosmetic I being $42.00 and Cosmetic II being $69.00. Therefore, the final answer is {"answer": "Yes"}.

In: Here is the description of Cosmetic I. |Description|Price|Color| |---|---|---| |Dior glow maximizer primer|$42.00|1|

Cosmetic II: |Price|Color|Description| |---|---|---| |$43.91|1|Dior Glow Maximizer Light Boosting Primer|

Out: Cosmetic I and Cosmetic II are the same product, Dior Glow Maximizer Light Boosting Primer. Both have the same

price ($42.00 and $43.91) and color (1). The only difference is the order of the columns. Therefore, the final answer is {"answer": "Yes"}.

In: Here is the description of Cosmetic I. |Description|Price|Color| |---|---|---| |Bobbi Brown foundation stick|$44.00|Warm Honey|

Cosmetic II: |Price|Color|Description| |---|---|---| |$58.50|Warm Ivory|Exclusive Make Up Product By Bobbi Brown Foundation Stick - #1 Warm Ivory 9g/0.31oz|

Let's think step by step and show your reasoning before showing the final result. Return the final result as JSON in the format {"answer": "<Yes or No>"}.

Out: Completion: Cosmetic I and Cosmetic II are different entities because they have different prices, colors, and descriptions.

Cosmetic I is a Bobbi Brown foundation stick that costs $44.00 and is in the color Warm Honey. Cosmetic II is an Exclusive Make Up Product By Bobbi Brown Foundation Stick that costs $58.50 and is in the color Warm Ivory. Therefore, the final answer is {"answer": "No"}.

#### Data Imputation Zero-Shot

Prompt: Task: Please fill in the missing value in the input table. The missing value is denoted by '[MISSING]'. Please

only return the value filled in. Do not return the whole table. Return the final result as JSON in the format {"value": "<value filled in>"}.

In: |Title|Credited artist(s)|Year| |---|---|---| |"10 or a 2-Way"|Korn|2005| |"4 U"|Korn|1999| |"A Different World"|Korn Corey Taylor|2016| |"A.D.I.D.A.S."|[MISSING]|1996| |"Alive"|Korn|2003|

Return the final result as JSON in the format {"value": "<value filled in>"}. Out:

Completion: {"value": "Korn"}

#### Data Imputation Few-Shot

Prompt: Objective: To fulfill your responsibility, please provide the missing value in the input table, represented by

'[MISSING]'. Only the filled-in value should be provided, not the whole table. Return the final result as JSON in the format {"value": "<value filled in>"}.

Input: |Column1|area|id_ov| |---|---|---| |16|CRM|29| |25|CCS|15| |32|CRM|4| |12|CRM|9| |8|[MISSING]|9|

Output: {"value": "CRM"}

Input: |Column1|area|id_ov| |---|---|---| |0|CRM|5| |1|[MISSING]|9| |2|CCS|9| |3|CCS|9| |4|CRM|9|

Return the final result as JSON in the format {"value": "<value filled in>"}. Output:

Completion: {"value": "CCS"}

#### Row/Column Sorting Zero-Shot

Prompt: Description: Please sort the table by the column "Gauge" such that the values in this column are in

alphabetically ascending order. Please return the sorted table. Q:

|Gauge|Dia.|Delta| |---|---|---| |5/0|0.5|nan| |5/0|0.5|0.046| |4/0|0.454|0.046| |4/0|0.454|0.029|

- |3/0|0.425|0.029| A: Completion: |Gauge|Dia.|Delta| |---|---|---|

- |3/0|0.425|0.029|

- |4/0|0.454|0.046|

|4/0|0.454|0.029| |5/0|0.5|nan|

- |5/0|0.5|0.046|

#### Row/Column Sorting Few-Shot

Prompt: Instruction: Could you sort the table by arranging it based on column headers in alphabetical descending order

from left to right? After the sorting, kindly return the table.

Input: |Rank|State|%| |---|---|---| |46.0|NH|18.7| |45.0|MT|19.0| |49.0|NY|18.3| |48.0|OR|18.5| |51.0|TX|18.1|

Output: |State|Rank|%| |---|---|---| |NH|46.0|18.7| |MT|45.0|19.0| |NY|49.0|18.3| |OR|48.0|18.5| |TX|51.0|18.1|

Input: |Rank|State|%| |---|---|---| |40.0|KS|20.0| |41.0|GA|20.0| |42.0|ND|19.6| |43.0|VA|19.3| |44.0|RI|19.3|

Output: Completion: |State|Rank|%|

|---|---|---| |KS|40.0|20.0| |GA|41.0|20.0| |ND|42.0|19.6| |VA|43.0|19.3| |RI|44.0|19.3|

