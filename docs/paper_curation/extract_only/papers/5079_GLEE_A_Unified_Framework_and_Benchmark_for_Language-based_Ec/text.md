# arXiv:2410.05254v3[cs.CL]2Mar2026

## GLEE: A Unified Framework and Benchmark for Language-based Economic Environments

EILAM SHAPIRA∗, Technion - Israel Institute of Technology, Israel OMER MADMON, Technion - Israel Institute of Technology, Israel ITAMAR REINMAN, Technion - Israel Institute of Technology, Israel SAMUEL JOSEPH AMOUYAL, Tel Aviv University, Israel ROI REICHART, Technion - Israel Institute of Technology, Israel MOSHE TENNENHOLTZ, Technion - Israel Institute of Technology, Israel

Large Language Models (LLMs) show significant potential in economic and strategic interactions, where communication via natural language is often prevalent. This raises key questions: Do LLMs behave rationally? How do they perform compared to humans? Do they tend to reach an efficient and fair outcome? What is the role of natural language in strategic interaction? How do characteristics of the economic environment influence these dynamics? These questions become crucial concerning the economic and societal implications of integrating LLM-based agents into real-world data-driven systems, such as online retail platforms and recommender systems. While the ML community has been exploring the potential of LLMs in such multi-agent setups, varying assumptions, design choices and evaluation criteria across studies make it difficult to draw robust and meaningful conclusions. To address this, we introduce a benchmark for standardizing research on two-player, sequential, language-based games. Inspired by the economic literature, we define three base families of games with consistent parameterization, degrees of freedom and economic measures to evaluate agents’ performance (self-gain), as well as the game outcome (efficiency and fairness). We develop an open-source framework for interaction simulation and analysis, and utilize it to collect a dataset of LLM vs. LLM interactions across numerous game configurations and an additional dataset of human vs. LLM interactions. Through extensive experimentation, we demonstrate how our framework and dataset can be used to: (i) compare the behavior of LLM-based agents in various economic contexts; (ii) evaluate agents in both individual and collective performance measures; and (iii) quantify the effect of the economic characteristics of the environments on the behavior of agents. Our results suggest that the market parameters, as well as the choice of the LLMs, tend to have complex and interdependent effects on the economic outcome, which calls for careful design and analysis of the language-based economic ecosystem.

### 1 Introduction

Recent research has increasingly focused on the capabilities of Large Language Models (LLMs) in decision-making tasks, revealing their potential to operate as autonomous agents in various economic environments, that typically require involving complex strategic thinking (Horton 2023; C. Li et al. 2024; H. Sun et al. 2025; G. Wang et al. 2023; Zhu et al. 2024). Applications of LLM-based agents in such environments include Chess-playing (Feng et al. 2024), task-oriented dialogue handling (Ulmer et al. 2024), financial advisement (Lakkaraju et al. 2023) and beyond. The promising capabilities of LLMs in strategic decision-making call for the study of these agents’ behavior through the lens of game theory, e.g. whether and when LLM-based agents naturally converge to Nash-equilibrium (Guo et al. 2024), and how well they learn to cooperate in repeated interactions (Akata et al. 2025). The rise of LLM

∗Corresponding Author.

Authors’ Contact Information: Eilam Shapira, orcid: 0009-0007-4536-5758, eilam.shapira@gmail.com, Technion - Israel Institute of Technology, Haifa, Israel; Omer Madmon, orcid: 0009-0001-4009-0368, omermadmon@gmail.com, Technion - Israel Institute of Technology, Haifa, Israel; Itamar Reinman, orcid: 0009-0003-4749-5004, itamarr@campus.technion.ac.il, Technion - Israel Institute of Technology, Haifa, Israel; Samuel Joseph Amouyal, orcid: 0009-0006-2068-2864, samamouyal201195@gmail.com, Tel Aviv University, Tel Aviv, Israel; Roi Reichart, orcid: 0000-0001-6918-0554, roireichart@gmail.com, Technion - Israel Institute of Technology, Haifa, Israel; Moshe Tennenholtz, orcid: 0000-0002-9459-5388, moshe.tennenholtz@gmail.com, Technion - Israel Institute of Technology, Haifa, Israel.

agents also calls for a clear benchmark for assessing the extent to which such agents behave rationally (Raman et al. 2024).

An important property of many real-world economic environments is that communication between agents typically occurs through natural language. While economic modeling usually abstracts away these nuances for the sake of keeping the model simple and tractable, in practical scenarios the fact that natural language is involved may significantly affect the interaction outcome. For instance, in bargaining between two parties, the same offer can be interpreted very differently depending on how it is framed. Consider the following two offers for a business partnership:

- (a) "We propose a 40-60 split as we believe your expertise will drive the majority of the success."
- (b) "We propose a 40-60 split because we’ve already invested significant resources and need a lower share to maintain balance."

Both of these offer the same numerical terms, but the framing in each case may lead to different reactions based on how the rationale is presented. A similar rationale applies to a broader range of economic scenarios. To illustrate this further, let us consider the following representative use cases:

- • Bargaining. Alice and Bob co-own a startup valued at a million dollar, and must decide how to divide the proceeds from its sale. To reach an agreement, they take turns proposing how to split the shared value, with each delay in reaching a decision reducing the overall value for both parties. The interaction often involves free language in the proposal exchange.
- • Negotiation. Alice aims to sell a product to a potential buyer, Bob. They negotiate by taking turns proposing prices, with Alice deciding whether to accept Bob’s offer and sell, and Bob deciding whether to accept Alice’s price and buy, continuing this process until they reach an agreement or the negotiation concludes. The negotiation often involves natural language, such as descriptions of the product, which can influence the outcome.1
- • Persuasion. Alice is a seller trying to sell a product to Bob at an exogenously set fixed price. Alice knows the true quality of the product, while Bob only has a rough expectation of the product’s quality. Alice sends a message to persuade Bob to buy the product, aiming to convince him of its value, regardless of its true quality. Bob, however, benefits only if the product is of genuinely high quality. The interaction then repeats for multiple rounds, with the product quality in each round being independently drawn. In such a scenario, Alice balances between her maximizing one-shot gain and building positive reputation, and linguistic communication can play a significant role.

These three types of interactions are inspired by influential models in the economics literature. They are also broad and flexible enough to capture a wide variety of real-world applications. The bargaining model, inspired by the seminal work of Rubinstein (1982), forms the basis for understanding how parties negotiate the division of shared resources, applicable in scenarios such as business mergers, partnership dissolution, and legal settlements. The negotiation game reflects celebrated models of bilateral trade (McAfee 1992; Myerson and Satterthwaite 1983) where buyers and sellers negotiate over the prices of an indivisible good, which often arises in various applications, including real estate transactions, corporate acquisitions, and e-commerce platforms. Lastly, the persuasion model draws on classical models of information asymmetry and strategic communication (Akerlof 1978; Crawford and Sobel 1982; Farrell and Rabin 1996), which play a pivotal role in advertising, marketing, political campaigning, and recommendation systems.

A key feature of all these games is that they are sequential, meaning players take turns acting rather than acting simultaneously. This makes communication, and in particular language-based communication, a crucial

1Notice that in the economic literature, the terms "bargaining" and "negotiation" are often used interchangeably to describe both the division of a divisible good and the process of price negotiation. In this paper, we use the terminology "bargaining" to describe the former and "negotiation" for the latter, for ease of exposition.

element of the interaction. In sequential games, communication is typically direct, with players able to attach messages to their actions (e.g., in the bargaining example, the textual message is attached to the proposal). In contrast, in simultaneous games, communication tends to be indirect, as players act independently and typically learn to cooperate by observing and reacting to past actions rather than through direct message exchange. A prime example of a simultaneous, language-based economic interaction is the competition among web publishers in search engines. Publishers simultaneously create content (usually in the form of textual documents) for their websites, and then gradually learn and adapt based on outcomes, such as their relative search engine rankings and exposure. However, they usually lack the opportunity for real-time communication or coordination during this process. These simultaneous, language-based interactions are well-studied within the information retrieval community, and the induced publishers’ game is indeed modeled as a simultaneous game.2 Our work is complementary, focusing on sequential settings where direct, language-based communication is prevalent.

The AI and NLP communities have recognized the potential of integrating natural language into stylized economic models of bargaining, negotiation and persuasion. They also recognize the importance of studying the capabilities and limitations of LLM-based agents within such frameworks. Research increasingly focuses on evaluating and improving LLM agents in bargaining and negotiation settings (Abdelnabi et al. 2023; Bianchi et al. 2024; Deng et al. 2024; Hua et al. 2024; Noh and Chang 2024; Xia et al. 2024). In the realm of persuasion games, recent studies have introduced language-based frameworks that explore the design of optimal information transmission policies (M. Raifer et al. 2022) and develop methods for generating data to predict human players’ behavior (Apel et al. 2022; Shapira, Madmon, Apel, et al. 2025; Shapira, Madmon, Reichart, et al. 2024).

Although each of these economic setups has been studied in the context of LLM-based agents, the approaches have varied widely across different studies. Each paper often adopts distinct modeling and implementation assumptions, poses unique research questions, and applies diverse evaluation criteria, making it challenging to compare results and draw broader conclusions about the capabilities and limitations of LLM-based agents in economic environments. Moreover, the vast majority of the existing game-theoretic benchmarks for LLMs do not even include setups in which agents are allowed to communicate in natural language (Duan et al. 2024; Guo et al. 2024; Huang et al. 2024; H. Wang et al. 2024).

To enhance the real-world reliability of LLM-based agents, it is essential to establish a clear benchmark that provides a standardized framework for modeling these economic interactions. This benchmark would ensure comparability across studies and enable generalization of findings, facilitating a deeper understanding of how various factors influence interaction outcomes and leading to more robust and reliable conclusions about the performance of LLM-based agents in real-life economic situations.

### 1.1 Our Contribution

We introduce GLEE, a unified framework for Games in Language-based Economic Environments,3 focusing on the case of two-player games. The two-player case is central, as it marks the shift from single-agent decision-making to strategic, communicative interaction. Such scenarios are common in real life, intuitive to analyze, and still capture the essential dynamics of more complex multi-agent settings. Our framework provides a principled way to evaluate the performance of LLM agents and human players in a wide class of fundamental language-based economic scenarios.

Games Parametrization. Central to the framework is a clear and comprehensive parameterization of the space of all bargaining, negotiation and persuasion games (as described above). It defines degrees of freedom and evaluation metrics that are consistent across economic contexts. The generality of the space of games spanned

2For instance, see Ben-Porat et al. (2019), Hron et al. (2022), Kurland and Tennenholtz (2022), Madmon et al. (2025a,b), Nachimovsky et al.

(2024), and N. Raifer et al. (2017), in which SEO competitions are modeled as simultaneous games. 3Code and data are available in https://github.com/eilamshapira/GLEE.

by our parameterization follows from the richness of the degrees of freedom considered, which include the game horizon (number of rounds within each game), information structure (whether agents know each others’ preferences or not), and communication form (whether agents communicate via free language or structured messages). The framework is implemented as an open-source codebase that allows researchers to instantiate a wide range of economic games and evaluate LLM behavior within them. While the specific LLMs used in experiments may evolve or become outdated, the evaluation setup and metrics remain valid.

Data Collection. Using this framework, we have collected a dataset of LLM vs. LLM games, comprising 587K decisions made by LLM-based agents across more than 80K games, involving 13 different LLMs. The framework enables controlled experimentation across a wide range of language models, game types, and experimental conditions, making the dataset a valuable resource for evaluating LLM behavior in diverse economic settings. It also supports in-depth analyses of how the parameters of the economic environment affect agent strategies and shape interaction outcomes. Additionally, we developed an interactive interface that enables human participants to compete against LLM-based agents across various game configurations. Using this setup, we conducted experiments and compiled a complementary dataset of human vs. LLM interactions.

Experimental Evaluation. Through extensive analysis of the collected data, we uncover meaningful behavioral and economic phenomena:

- (1) We find that the economic outcome, measured by efficiency, fairness, and agent self-gain, is significantly shaped by market parameters, including the information structure, communication form, and interaction horizon.
- (2) We show that there are no absolute best-performing models in terms of any of the evaluation measures, and that the performance of one LLM strongly depends on its competitor’s choice of LLM.
- (3) We compare LLMs to human participants and find that human behavior is markedly more extreme: in each setting, they either outperform all LLMs or fall behind them entirely, depending on the game and their assigned role.

Together, these findings show how GLEE can be used not only as an infrastructure for research, but also as a tool for deriving concrete insights into economic reasoning and strategic behavior in language-based interactions.

### 1.2 Comparison to Prior Benchmarks

To put in context the contribution of GLEE within the broader research landscape, we compare it to several recent benchmarks designed to evaluate the performance of LLMs in strategic settings. While these benchmarks differ in their choice of games, evaluation metrics, and interaction modalities, none focus on multi-turn economic games involving natural language communication. GLEE addresses this gap by offering a unified, richly parameterized framework for studying LLMs in sequential, language-based economic interactions. The following comparisons highlight where GLEE introduces new capabilities and how it complements or extends existing benchmarks.

LLMs as Rational Players. (C. Fan et al. 2023). Test the rationality of LLM in the light of game theory, using three simple one-shot games with no interaction between the players. GLEE supplies multi-round negotiation/bargaining/persuasion with free-text chat and direct human–agent comparison.

GAMA-Bench. (Huang et al. 2024). Introduce 8 single-turn games studied in game theory to test LLM rationality. The games are not economic games with little interaction between players. GLEE targets language–centric games, with games extensively studied in economic literature and extensive human trials.

GTBench. (Duan et al. 2024). Introduces a benchmark to test LLM rationality using board and card games. These games are not economic games, and natural language interaction does not happen. GLEE emphasis is on natural language communication (e.g., concessions, persuasion) within purely economic games.

Game theoretic LLM. (Hua et al. 2024). Measures LLM behaviour on different multi-turn non-economic games extensively studied in game theory. The games they use have no extensive parameterization, and therefore few variants. GLEE is based on three economic games with heavy parameterization, allowing testing LLMs on hundreds of variants of the same game.

TMGBench. (H. Wang et al. 2024). Introduces an exhaustive benchmark with 144 2 × 2 normal-form games, but these games are not economic games, are often synthetic, single-shot, and without communication between the players. While GLEE covers fewer games, it captures real dynamics with multi-turn and natural language communication between the player.

CompeteAI. (Zhao et al. 2023). Introduce a one-competition world with multiple players; the metrics used are qualitative (LLM judgements). GLEE generalises across three economic game families with multiple parametrizations, using quantitative metrics derived from game theory research.

We summarize key differences in Table 1, highlighting where GLEE introduces new capabilities.

- Table 1. Comparison of benchmarks evaluating LLMs in strategic games. GLEE stands out by focusing on multi-turn, natural language, economic interactions with extensive human evaluation and openness.

###### Framework Multi turns NL Communication Human Evaluation

GLEE (ours) ✓ ✓ ✓ Rational Players (C. Fan et al. 2023) ✗ ✗ ✗ GAMA-Bench (Huang et al. 2024) ✗ ✗ ✓ GTBench (Duan et al. 2024) ✓ ✗ ✗ Game theoric LLM (Hua et al. 2024) ✓ Structured ✗ TMGBench (H. Wang et al. 2024) ✗ ✗ ✗ CompeteAI (Zhao et al. 2023) ✓ Indirect ✗

### 2 Game Families and Parametrization

In this section, we formally define the three game families discussed in the introduction: bargaining, negotiation and persuasion. We then further discuss the related economic literature, and review some well-known theoretical results concerning these games. For each family, we provide a formal game-theoretic definition of the game (including the players, strategies and utilities), define degrees of freedom, and define the game outcome and evaluation metrics. While previous research has effectively tackled the challenge of evaluating the rationality of individual agents (Raman et al. 2024), our focus extends to evaluating the outcome reached by the strategic behavior of the agents, using the fundamental economic notions of efficiency and fairness.4

In our modeling, we focus on three main market characteristics that are critical for understanding the dynamics of LLM-based agents in economic interactions: game horizon, information structure, and communication form. The game horizon refers to the number of time periods during which the game is played and whether the length of the horizon is known or unknown to the agents.5 This factor influences the strategies agents adopt, particularly in terms of long-term planning and anticipation of future moves. The information structure determines whether agents are aware of each other’s preferences, impacting their ability to predict and respond to the actions of

4Metrics are normalized to [0, 1], such that higher values indicate better (more efficient or fair) outcomes. 5In economic theory, "infinite horizon" typically refers to a large, unspecified duration. Accordingly, we use the term "infinite horizon" to describe cases where the horizon is both large and unknown to the agents.

others. Lastly, the communication form specifies whether communication between agents occurs through free language or structured, concise messages, which affects the richness and clarity of the exchanges.

### 2.1 Bargaining Games

The first family of games is inspired by the celebrated bargaining model of Rubinstein (1982). The model encompasses a class of bargaining games where two players, Alice and Bob, alternate offers over a time horizon 𝑇 (usually, 𝑇 = ∞) to divide a fixed sum of money 𝑀 between them. It is clear that under a full rationality assumption, the amount of money does not play a role in the analysis. However, for human (or LLM) players, it is evident that the amount of money indeed matters. Importantly, in these games delays are costly, a concept captured by discount factors 𝛿𝐴,𝛿𝐵 assigned to each player, reflecting the decreasing value of future payoffs as time progresses. Formally:

- (1) At each odd stage 𝑡, Alice offers a division (𝑝, 1 − 𝑝) for some 𝑝 ∈ [0, 1].
- (2) Bob decides whether to accept or reject.
- (3) If Bob accepts, the (Alice, Bob) utility vector is given by 𝑀(𝛿𝐴𝑡−1𝑝,𝛿𝐵𝑡−1(1 − 𝑝)).
- (4) At each even stage 𝑡, Bob offers a division (𝑞, 1 − 𝑞) for some 𝑞 ∈ [0, 1].
- (5) Alice decides whether to accept or reject.
- (6) If Alice accepts, the game terminates, and the (Alice, Bob) utility vector is given by 𝑀(𝛿𝐴𝑡−1𝑞,𝛿𝐵𝑡−1(1 − 𝑞)).

If no agreement is reached at any stage, the utilities are defined to be (0, 0). In the standard version of the game, the time horizon is infinite and the discount factors are common knowledge (i.e., Alice and Bob know both 𝛿𝐴 and 𝛿𝐵).

In our experiments, we simulate a wide range of such bargaining games, differing in the following degrees of freedom:

- (i) whether or not the players know their opponent’s discount factor (complete vs. incomplete information);
- (ii) whether or not the players know when the game terminates (finite vs. infinite);
- (iii) whether or not players’ communication involve natural language (structured vs. linguistic);
- (iv) the values of 𝛿𝐴,𝛿𝐵 ∈ (0, 1) and 𝑀;
- (v) the value of 𝑇 in the finite horizon case.

An outcome of the game is a pair (𝑡𝑒𝑣,𝑝𝑒𝑣), where 𝑡𝑒𝑣 is the stage index at which the game terminated, and 𝑝𝑒𝑣 is the share of 𝑀 that Alice obtained (without considering the discount in the utilities). When the game terminates without reaching an agreement, we define 𝑡𝑒𝑣 = ∞, and the gain for both players is zero. The evaluation metrics used to assess the economic outcome are efficiency and fairness. Efficiency is now measured by the normalized sum of Alice’s and Bob’s discounted payoffs at the time of agreement:

𝛿𝐴𝑡𝑒𝑣−1𝑝𝑒𝑣 + 𝛿𝐵𝑡𝑒𝑣−1(1 − 𝑝𝑒𝑣)

- if 𝑡𝑒𝑣 < ∞, and 0 if 𝑡𝑒𝑣 = ∞. Fairness is defined as the distance between the actual division and the fairest division, 1 − 4 · (𝑝𝑒𝑣 −

- 1

- 2)2

- if 𝑡𝑒𝑣 < ∞, and 1 if 𝑡𝑒𝑣 = ∞. We consider the case of no trade as fair, since both players get the same utility. Obviously, this is also the least efficient outcome, which highlights the natural fairness-efficiency tradeoff.

#### 2.2 Negotiation Games In the second family of games, referred to as negotiation games, a seller (Alice) and a buyer (Bob) are negotiating over the price of a product. At the outset, Alice owns a product she subjectively values at 𝑉𝐴. The subjective valuation of the potential buyer, Bob, is 𝑉𝐵. To capture the notion of valuation scale in negotiation games, we parameterize 𝑉𝑖 = 𝑀 · 𝐹𝑖 for 𝑖 ∈ {𝐴,𝐵}, where 𝐹𝑖 ∈ (0, 1) is a factor parameter 𝑀 is a scale parameter.

As in the case of bargaining games, Alice and Bob alternate offers: at each odd stage, Alice posts a price and Bob decides whether to buy the product or move on to the next stage. At each even stage, Bob is the one to post a price and Alice decides whether to sell at this price or reject and move on to the next stage. The game is played for 𝑇 stages, which again can be either finite or "infinite" (i.e., large and unknown to both players). Unlike the bargaining game, here we assume no discount factors on the utilities, hence whenever a price 𝑝 is accepted, the utilities for Alice and Bob are 𝑝 −𝑉𝐴 and 𝑉𝐵 − 𝑝 respectively. If no trade is made, then the utilities are defined to be (0, 0).

In this class of games, we consider the following degrees of freedom:

- (i) whether or not the players know their opponent’s product valuation (complete vs. incomplete information);
- (ii) whether or not the players know when the game terminates (finite vs. infinite);
- (iii) whether or not players’ communication involve natural language (structured vs. linguistic);
- (iv) the values of 𝐹𝐴, 𝐹𝐵 ∈ (0, 1) and 𝑀; and
- (v) the value of 𝑇 in the finite horizon case.

An outcome of a negotiation game is captured by 𝑝𝑒𝑣, which is the price at which the product is sold when there is a trade, and defined to be 𝑝𝑒𝑣 = ∅ whenever no trade is made. We consider the following evaluation measures of the game outcome fairness and efficiency. When there is trade, fairness is measured by

2

𝑝𝑒𝑣 − 𝑝𝑓 𝑀

1 − 4 ·

where 𝑝𝑓 = 𝑉𝐴+2𝑉𝐵 is the "fairest price".6 When trade is not made, we define the fairness to be 1 (i.e., maximal fairness) to reflect that no-trade does not change the default allocation of the product.

Efficiency is defined to be 1 in the following cases (and zero otherwise):

- (a) Alice values the product more than Bob, and does not sell it (𝑉𝐴 ≥ 𝑉𝐵 and 𝑝 = ∅); and
- (b) Alice values the product less Bob, and sells the product at a price that is beneficial for both players (𝑉𝐴 ≤ 𝑝 ≤ 𝑉𝐵).

When averaged over a large number of simulated games for a certain game configuration, this measure estimated the probability of the event "an efficient trade occurs when it should occur".

### 2.3 Persuasion Games

In a persuasion game, a seller (Alice) tried to persuade a buyer (Bob) to buy a product at a fixed price 𝜋. Alice privately knows the true product quality, which can be either high or low. Bob only knows that the prior probability of the product being of high quality is 𝑝. Alice gets a utility of 1 if Bob buys (regardless of the true product quality), and 0 otherwise.

Bob values a high-quality product at 𝑣 > 𝜋 and a low-quality product at 𝑢 < 𝜋. Therefore, the utility of Bob from buying a high-quality product is 𝑣 − 𝜋 > 0, and from buying a low-quality product is 𝑢 − 𝜋 < 0. For simplicity, we normalize the price to 𝜋 = 1 and the value of a low-quality product to 𝑢 = 0. In addition, we consider a currency scale parameter 𝑀 that serves as a multiplicative term of Bob’s utility function. Overall Bob gets a utility of 𝑀(𝑣 − 1) from buying a high-quality product, −𝑀 from buying a low-quality product, and 0 from not buying the product. The timing of a single round is as follows.

(1) First, Alice observes the product quality (which is realized to be high-quality w.p. 𝑝, independently of other rounds).

6Notice that unlike all other metrics, the fairness metric in negotiation games is not normalized, as the price 𝑝𝑒𝑣 is unbounded in principle. However, we observe that normalizing the difference 𝑝𝑒𝑣 − 𝑝𝑓 by the scale parameter 𝑀 results in a measure that is between 0 and 1 in 99.5% of the cases.

(2) Then, Alice sends a message to Bob. Lastly, Bob decides whether to buy or not to buy the product, and utilities are realized accordingly.

The game then consists of 𝑇 such rounds. Alice’s goal is to maximize her cumulative utility over time. We differentiate between two types of persuasion game setups:

- (a) Long-living buyer. The buyer is long-living, in the sense that he also aims to gain his cumulative utility over time. In this case, both players observe the entire interaction history.
- (b) Myopic buyers. Buyers are myopic, in the sense the buyer of stage 𝑡 only cares about the utility obtained at stage 𝑡. In this case, each buyer observes the statistics of all previous rounds (i.e., % of rounds in which Bob bought the product, and % of rounds in which Bob bought a low-quality product). That is, each buyer observes sufficient statistics from the entire history. This implementation detail is due to context length memory which is an inherent limitation of LLM agents, as well as to reducing cognitive load on human players.

We consider the following degrees of freedom in persuasion games:

- (i) whether or not Alice knows Bob’s high-quality product valuation 𝑣 (complete vs. incomplete information);
- (ii) whether or not the players know when the game terminates (finite vs. infinite);
- (iii) whether or not Alice’s messages involve natural language (structured vs. linguistic);
- (iv) the values of 𝑣, 𝑝 and 𝑀;
- (v) the value of 𝑇 in the finite horizon case;
- (vi) whether the game is with a long-living Bob or with myopic buyers.

An outcome of the game is a tuple (𝑛𝑒𝑣,𝑘𝑒𝑣,𝑟𝑒𝑣), where 𝑛𝑒𝑣 is the number of rounds in which the product was of high-quality, 𝑘𝑒𝑣 is the number of rounds in which the product was of high-quality and the buyer bought the product, and 𝑟𝑒𝑣 is the number of rounds in which the product was of low quality and the buyer did not buy the product. We define efficiency to by the proportion of rounds in which the product was sold out of all rounds in which the product was of high quality (i.e., 𝑘𝑒𝑣

𝑛𝑒𝑣 ), and fairness to be the proportion of rounds in which the product was not sold out of all rounds in which the product was of low quality (i.e., 𝑟𝑒𝑣

𝑇−𝑛𝑒𝑣 ).

### 2.4 Game Families Through the Lens of Economic Literature

In this section, we discuss the economic literature related to the three game families considered in this paper, and review some known theoretical results.

Bargaining. The standard bargaining model of (Rubinstein 1982) consists of two players, Alice and Bob, engaging in alternating offers for a finite horizon (𝑇 = ∞) with commonly known discount factors 𝛿𝐴,𝛿𝐵. Our parameterization considers several additional degrees of freedom, such as finite vs. infinite time horizon, complete vs. incomplete information, and free language messages vs. structured and concise messages. In the standard model, Rubinstein (1982) showed that in the unique subgame-perfect equilibrium an agreement is reached in the first stage (i.e., utilities are not discounted), and the share of Alice is given by 𝑀𝑝∗, where 𝑝∗ = 1−1−𝛿𝛿𝐵

𝐴𝛿𝐵 .7 The case of a finite horizon can be solved using backward induction, and typically results in a different outcome compared to the infinite case. As𝑇 grows, the equilibrium outcome approaches the one of the infinite case. Extensions to incomplete information regarding the opponent’s discount factor are typically more challenging, and some of

- them are studied in the literature (Rubinstein 1985).

7A subgame-perfect equilibrium is a strategy profile in which every player responds optimally in every hypothetical subgame of the game, including off-path scenarios that are not reached in practice. This solution concept can be seen as capturing a higher level of rationality compared to the alternative of Nash equilibrium.

Negotiation. In a negotiation game, Alice and Bob negotiate over the price of an indivisible good. The negotiation game differs from the bargaining games in several key aspects:

- (1) Negotiation involves an indivisible product (e.g., Alice’s product), while bargaining focuses on dividing a divisible resource, such as money.
- (2) In negotiation, Alice and Bob may have different subjective valuations of the product, whereas in bargaining, both parties value the divisible resource similarly.
- (3) Negotiation has no discounting, so the utility remains constant over time. In bargaining, delays reduce the total value, encouraging faster agreement.

If the seller is uncertain regarding the buyer’s valuation but has a prior belief distribution, a classical result by Harris and Raviv (1981) and Riley and Zeckhauser (1983) states that it is always optimal to sell the product at a fixed price. In contrast, if the seller does not have a prior belief over the buyer’s valuation, and instead aims to minimize regret, then an optional pricing policy will be to randomly choose a price from a carefully chosen distribution (Bergemann and Schlag 2011).

Persuasion. Our persuasion game follows the structure of a cheap talk game (Crawford and Sobel 1982; Farrell and Rabin 1996), where the sender (Alice) cannot commit to a signaling policy in advance, unlike in Bayesian persuasion models (Kamenica and Gentzkow 2011). Under the particular payoff structure considered in our persuasion game, it is well-known that the cheap-talk game only admits a babbling equilibrium, i.e., an equilibrium in which all information is kept hidden (this is due to the strong misalignment of interests between the two players). In contrast, if the seller can commit to a signaling policy at the outset, as in Bayesian persuasion, then there exists a subgame-perfect equilibrium in which the seller commits to the following policy: When the product is of high quality, the seller recommends buying the product with probability 1. When the product is of low quality,

- then the seller recommends with probability 𝑞 = min{1−𝑝𝑝 (𝑣 − 1), 1}. This policy is also incentive-compatible, in the sense that the buyer always buys the product when the seller recommends buying.8 While the long-living buyer case is well-studied in the economic literature, such games often admit multiple equilibria, which makes the games difficult to analyze and predict (Aumann and Hart 2003; Kim 1996). As for the case of myopic buyers, (Best and Quigley 2024) draws a connection between the repeated cheap talk game and the case of one-shot Bayesian persuasion, leading to an elegant analytical solution of the repeated game. Intuitively, the repetitive nature of the game induces a reputation effect, which plays a similar role to the commitment power in standard one-shot Bayesian persuasion.

### 3 Data Collection and Statistical Analysis

In this section, we describe the process of data collection and analysis. We developed a user-friendly game management system to facilitate data collection from the games described in §2. The system is written in Python, designed for ease of use and customization, enabling future researchers to seamlessly collect data. Integrating new language models is straightforward, allowing them to participate in any configurable setup. The interface allows users to effortlessly run data collection across multiple configurations while involving various LLMs in the process. Additionally, the system supports analyzing the collected data to gain insights into the performance and interaction patterns of different models. The system features a simple and intuitive interface, as illustrated in the screenshots provided in Appendix A.1. Prompt samples are described in Appendix A.2.

Configurations. Since the game space defined by the parameters presented in §2 is infinite for each of the game families, it is clear that data cannot be collected from all possible games. Therefore, we attempted to cover the game space by selecting diverse values for each of the parameters defining the games, and we collected data from

8In fact, the probability of lying 𝑞 is determined such that the buyer is indifferent upon receiving a recommendation, taking into account his belief updating, which relies on using Bayes’s law and knowing the seller’s committed policy.

- Table 2. Parameters and their optional values used to define the 1,320 game configurations across bargaining, negotiation, and persuasion game families for data collection.𝑇 = ∞ means a very large value of𝑇, unknown to the players. CI = Complete Information. MA = Textual messages allowed.

##### Bargaining Negotiation Persuasion

𝛿𝐴 0.8, 0.9, 0.95, 1 𝐹𝐴 0.8, 1, 1.2, 1.5 p 13, 0.5, 0.8 𝛿𝐵 0.8, 0.9, 0.95, 1 𝐹𝐵 0.8, 1, 1.2, 1.5 v 1.2, 1.25, 2, 3, 4 𝑀 102, 104, 106 𝑀 102, 104, 106 M 102, 104, 106 𝑇 12, ∞ 𝑇 1, 10, ∞ 𝑇 20 CI True, False CI True, False CI True, False MA True, False MA True, False Messages type Binary, Textual

Buyer type Long-living, Myopic In total 384 configurations In total 576 configurations In total 360 configurations

every possible combination generated by these parameters. Table 2 shows the parameters defining the groups from which we collected data. In total, we collected data from 1,320 different configurations: 384 configurations of bargaining, 576 configurations of negotiation, and 360 configurations of persuasion games.

Data. For data collection, we utilized 13 state-of-the-art large LLMs, spanning multiple architectures and vendors. Table 3 presents the 13 LLMs used for data collection, organized by model name, developer, and release year. The 169 possible pairs (including a language model playing against itself, with attention to the assignment for Alice or Bob) played across 1,320 different configurations. Each of the LLMs played as Alice in 2,500 bargaining games, 2,500 negotiation games, and 1,000 persuasion games, and an equal number of games as Bob. Table 4 contains statistics on the data we collected, which constitutes a contribution of the paper. To enable comparison with human behavior, we developed an interface for collecting data from games played between LLMs and humans, and gathered data from 3,405 human participants. Details about the human data collection process are provided in Appendix B.

Table 3. Large Language Models used for data collection.

Developer Model Name Release Year Anthropic

Claude 3.5 Sonnet 2024 Claude 3.7 Sonnet 2024

- Gemini 1.5 Flash 2024

- Gemini 1.5 Pro 2024
- Gemini 2.0 Flash 2025

- Gemini 2.0 Flash Lite 2025

Google DeepMind

GPT-4o 2024 GPT-4o Mini 2024 GPT-O3 Mini 2024

OpenAI

xAI Grok 2 2024 Meta

LLaMA 3.1–405B 2024 LLaMA 3.3–70B 2024

Mistral AI Mistral Large 2024

Table 4. Statistics of data collected by family.

Game Type Games Decisions Messages Words Bargaining 33.7K 92.8K 45.8K 1.14M Persuasion 13.5K 402K 140K 6.95M Negotiation 32.9K 92.3K 44.4K 1.69M Total 80.1K 587K 230K 9.77M

Statistical Analysis. To enable adequate comparisons between models that played in different game configurations, we employ a statistical framework that controls for variation in game structure and player composition. Since each model encountered only a subset of all possible game setups, raw metric values are not directly comparable. To address this, we fit linear regression models that predict each target metric based on the full parameterization of the game and the participating players. This allows us to estimate the independent contribution of each parameter while accounting for confounding effects introduced by the game configuration. In doing so, we obtain normalized estimates of model behavior that can be compared across heterogeneous settings. We train a separate linear regression model for each combination of game family and evaluation metric (introduced in §2). The construction of the feature representation is detailed in Appendix C.1.

The estimated regression coefficients (𝛽 values) enable us to quantify the impact of each parameter value on the metric relative to a predetermined default value. The list of default values is detailed in Appendix C.2. In addition to the magnitude of the effects, we also report 95% confidence intervals, computed using the standard procedure for linear regression coefficients (Wooldridge 2016). Appendix C.3 shows that the linear model performs comparably to state-of-the-art regression models on our prediction tasks, justifying its use in our analyses.

Importantly, metric values presented in this section are calculated by averaging over all possible game configurations in our dataset. These averages are therefore highly sensitive to the particular configurations and the configuration distributions in our dataset. To tailor the benchmark to specific applications, we recommend re-defining the parameter space and their distributions according to the economic context.9

### 4 Economic and Behavioral Insights

In this section, we present the findings of our analysis, focusing on how different configurations and player compositions affect the outcomes of the games. We examine the general trends observed across all game families and identify specific characteristics that differentiate the performances of language models and human players. We structure our results around the following research questions:

- (Q1) How do the market characteristics, such as information, game horizon, and linguistic communication, affect efficiency and fairness?
- (Q2) How do different LLMs behave in strategic interactions, and which models achieve fair, efficient, and high self-gain outcomes?
- (Q3) How do humans perform compared to LLMs?

Influence of Game Parameters (Q1). Table 5 presents the effect of the market-defining parameters on fairness and efficiency, respectively, for each of the game families. From the table, we observe that market conditions have a decisive impact on the fairness and efficiency achieved in the game.

9For instance, one could ask whether agents’ performance significantly differs in economic environments where inflation is high (translating into lower discount factors, in bargaining games). To evaluate such a scenario, one can simulate games in which the discount factor distribution fits these conditions, and re-evaluate agents’ performance with respect to the new distribution of configurations.

Table 5. The effect of the market-defining parameters on the efficiency and the fairness of the game, measured in percentage point improvement relative to the naive parameterization. CI = Complete Information; MA = Textual Messages Allowed; MY=Myopic Buyers; Eff. = Efficiency, Fair. = Fairness; Conf. Int. = Confidence Interval. Bolded values indicate the highest metric values within the confidence interval, while underlined values indicate the lowest metric values.

(a) Bargaining CI MA T Eff. Fair.

- – ✓ ∞ 2.8 4.4
- – – ∞ 2.2 2.8

✓ ✓ ∞ 1.6 0.9

✓ – ∞

- – ✓ 12 -0.2 4.4
- – – 12 -0.8 3

✓ ✓ 12 -1.8 0.9

✓ – 12 -3.7 -0.8 Conf. Int. ∈ ±0.5 ±0.6

(b) Negotiation CI MA T Eff. Fair.

✓ – ∞ 17.4 0.2 ✓ ✓ ∞ 17.3 0.3 ✓ – 10 16.7 0.1 ✓ ✓ 10 16.3 0.1

- – – 10 9.2 -1.2
- – ✓ 10 8.3 -2.2

✓ – 1 6.8 -0.9 ✓ ✓ 1 5.5 -0.3

- – – ∞ 5.5 -0.5
- – ✓ ∞ 4.1 -1.4
- – – 1
- – ✓ 1 -2 0.2 Conf. Int. ∈ ±1.7 ±0.3

(c) Persuasion CI MA MY Eff. Fair.

##### ✓ – – (max) (max)

- – – – -3.2 -13.4

✓ – ✓ -5.7 -0.7

- – – ✓ -10.9 -6.6
- – ✓ – -12.2 -15.6

✓ ✓ – -13.6 -12.7

- – ✓ ✓ -22.4 -22.2

✓ ✓ ✓ -24.7 -22.7 Conf. Int. ∈ ±2.5 ±2.4

Across all game families, a prolonged interaction between Alice and Bob consistently enhanced both efficiency and fairness. However, the effects of complete information and textual message allowance varied by game type. In bargaining games, complete information reduced both efficiency and fairness, while allowing messages improved both metrics. Conversely, in persuasion games, the pattern was nearly reversed: allowing messages degraded both efficiency and fairness. This held true for scenarios in which Bob was either a casual or a repeat buyer. In contrast, complete information improved fairness in both settings without significantly affecting efficiency. In negotiation games, complete information improved both metrics, whereas message allowance enhanced fairness without impacting efficiency.

We hypothesize that the impact of free language communication differs across game types according to the presence of ground truth. In persuasion games, an objective truth (e.g., product quality) allows agents to build trust through consistent behavior, making free-form language less essential and potentially harmful if it introduces noise. In contrast, bargaining and negotiation games lack an objective ground truth, so language becomes essential for coordination, signaling intentions, and achieving mutually beneficial outcomes.

Interestingly, our results suggest that the effect of complete information on fairness and efficiency may depend on whether message allowance is enabled. In particular, in bargaining games, complete information appears

- to reduce efficiency and fairness only when linguistic messages are allowed. This interaction is noteworthy because traditional economic models typically abstract away from natural language communication. Our findings suggest that some classic theoretical insights, derived under structured or language-free assumptions, may not be applicable when rich linguistic communication is present. While a comprehensive theoretical investigation is out of scope, we believe this is a promising direction for future work at the intersection of language and economic theory.

LLMs Performance (Q2). Table 6 presents the agents’ performance as Alice and as Bob in each of the game families in terms of self-gain. Nevertheless, the choice of which LLM to deploy in order to maximize utility is

more nuanced than simply selecting the model with the best overall performance. Figures 1, 2, and 3 present the payoffs of Alice (left) and Bob (right) in bargaining, negotiation, and persuasion games as a function of the identity of the models playing each role. Notably, when Bob is fixed to play with the LLM that yields the highest utility for him across the evaluated LLMs (claude-3.5-sonnet, see Table 6), Alice maximizes her payoff by selecting to play with mistral-large-2411. However, this model generally performs poorly when playing as Alice (ranked 11th out of 13).

Table 6. The effect of the Agent on the self gain, for each game family and role in the game.

Family Bargaining Negotiation Persuasion Model Alice Bob Alice Bob Alice Bob

human 6.6 ± 1.1 -17.1 ± 1 -26.6 ± 1.1 -14.5 ± 1.1 20.5 ± 4.1 54.8 ± 8.9 llama-3.3-70b 1.4 ± 0.7 0.7 ± 0.6 2.5 ± 0.6 4.2 ± 0.6 2.1 ± 2.4 11.2 ± 4.2 claude-3-5-sonnet 1.4 ± 0.7 2.6 ± 0.6 3 ± 0.6 4.5 ± 0.6 3.6 ± 2.4 12.1 ± 4.2 claude-3-7-sonnet 1.4 ± 0.7 2.1 ± 0.6 2.9 ± 0.6 4.4 ± 0.6 4 ± 2.4 11.3 ± 4.2 gpt-4o 1 ± 0.7 0.2 ± 0.6 2.8 ± 0.6 4.2 ± 0.6 4.3 ± 2.4 15.1 ± 4.2 gemini-2.0-flash 0.3 ± 0.7 -0.1 ± 0.6 3.5 ± 0.6 3.7 ± 0.6 -0.1 ± 2.4 21.2 ± 4.2 gemini-1.5-flash 0 ± 0 0 ± 0 0 ± 0 0 ± 0 0 ± 0 0 ± 0 grok-2-1212 -0.3 ± 0.7 0.5 ± 0.6 3 ± 0.7 4.4 ± 0.6 4.6 ± 2.4 11.2 ± 4.2 gpt-4o-mini -0.4 ± 0.7 -1.5 ± 0.6 2.1 ± 0.6 3.5 ± 0.6 0.1 ± 2.4 10.1 ± 4.2 llama-3.1-405b -0.9 ± 0.7 -0.2 ± 0.6 4 ± 0.6 4.7 ± 0.6 -0.1 ± 2.4 21 ± 4.2 mistral-large-2411 -1 ± 0.7 -1.3 ± 0.6 2.5 ± 0.6 5.3 ± 0.6 2.6 ± 2.4 28.3 ± 4.2

- gemini-1.5-pro -1 ± 0.7 -1.5 ± 0.6 2.4 ± 0.6 5 ± 0.6 6.3 ± 2.4 25.3 ± 4.2
- gemini-2.0-flash-lite -1.4 ± 0.7 -0.3 ± 0.6 2.9 ± 0.6 2.9 ± 0.6 1.1 ± 2.4 9.7 ± 4.2 o3-mini -6.9 ± 0.7 1.4 ± 0.6 2.4 ± 0.6 3.8 ± 0.6 3.4 ± 2.4 24.2 ± 4.2

One can think of a meta-game in which Alice and Bob (the users) choose LLMs to represent them in an economic game, aiming to maximize their expected utility (over all game parameter realizations). The strategy space consists of available LLMs, and players know only the game family, rather than its specific parameters. Under this setup, the only pure-strategy Nash equilibria are:

- • Bargaining: Alice selects claude-3-5-sonnet and Bob selects claude-3-7-sonnet;
- • Negotiation: Alice selects gemini-2.0-flash and Bob selects gemini-1.5-pro, or Alice selects llama-3.1-405b and Bob selects grok-2-1212;
- • Persuasion: Alice selects gemini-1.5-pro and Bob selects mistral-large-2411.

Figures 4, 5, and 6 show how the pair of models playing bargaining, negotiation, and persuasion games influenced the efficiency and fairness of the game. These tables suggest that there is no single model that maximizes both efficiency and fairness against all other models, and that compatibility between models plays an important role in shaping these metrics. Furthermore, across all game families, the scenario in which Alice and Bob employed the same LLM neither improved nor degraded the efficiency or fairness of the game, compared to the scenario in which they used different LLMs.

Human Performance (Q3). Human performance deviates significantly from that of LLMs across the evaluated tasks, in terms of self-gain. In persuasion games, humans consistently outperformed all LLMs by a substantial margin. However, in negotiation games, their self-gain was the lowest among all other LLMs. An intriguing pattern emerged in the bargaining games: while humans performed considerably worse than language models when playing the role of Bob, they significantly outperformed the models when playing as Alice, despite the

(a) Alice Gain 1 2 3 4 5 6 7 8 9 A B C D

- 1. claude-3-5-sonnet

- 2. claude-3-7-sonnet

- 3. gemini-1.5-flash

- 4. gemini-1.5-pro

- 5. gemini-2.0-flash

- 6. gemini-2.0-flash-lite

- 7. gpt-4o

- 8. gpt-4o-mini

- 9. grok-2-1212

- A. llama-3.1-405b

- B. llama-3.3-70b

- C. mistral-large-2411

- D. o3-mini

| | |
|---|---|

−0.13 −0.02 0.09

(b) Bob’s Gain 1 2 3 4 5 6 7 8 9 A B C D

- 1.

- 2.

- 3.

- 4.

- 5.

- 6.

- 7.

- 8.

- 9.

- A.

- B.

- C.

- D.

| | |
|---|---|

−0.1 0.02 0.15

- Fig. 1. The effect of the identities of the two players (rows: Alice, columns: Bob) on self-gain in Bargaining games, reported relative to the mean outcome.

(a) Alice Gain 1 2 3 4 5 6 7 8 9 A B C D

- 1. claude-3-5-sonnet

- 2. claude-3-7-sonnet

- 3. gemini-1.5-flash

- 4. gemini-1.5-pro

- 5. gemini-2.0-flash

- 6. gemini-2.0-flash-lite

- 7. gpt-4o

- 8. gpt-4o-mini

- 9. grok-2

- A. meta/llama-3.1-405b

- B. meta/llama-3.3-70b

- C. mistral-large

- D. o3-mini

| | |
|---|---|

−0.07 0 0.06

(b) Bob’s Gain 1 2 3 4 5 6 7 8 9 A B C D

- 1.

- 2.

- 3.

- 4.

- 5.

- 6.

- 7.

- 8.

- 9.

- A.

- B.

- C.

- D.

| | |
|---|---|

−0.09 −0.02 0.05

- Fig. 2. The effect of the identities of the two players (rows: Alice, columns: Bob) on self-gain in Negotiation games, reported relative to the mean outcome.

game’s ostensibly symmetric structure in terms of information. A plausible explanation for this phenomenon lies in a well-documented behavioral bias known as the anchoring effect (Tversky and Kahneman 1974), where the way information is presented can heavily influence human decision-making, even when the substantive content

(a) Alice Gain 1 2 3 4 5 6 7 8 9 A B C D

1. claude-3-5-sonnet 2. claude-3-7-sonnet 3. gemini-1.5-flash

- 4. gemini-1.5-pro

- 5. gemini-2.0-flash

- 6. gemini-2.0-flash-lite

- 7. gpt-4o

- 8. gpt-4o-mini

- 9. grok-2 A. meta/llama-3.1-405b B. meta/llama-3.3-70b C. mistral-large D. o3-mini

| | |
|---|---|

−0.19 0.04 0.26

(b) Bob’s Gain 1 2 3 4 5 6 7 8 9 A B C D

1. 2. 3. 4. 5. 6. 7. 8. 9. A. B. C. D.

| | |
|---|---|

−0.26 0.01 0.27

- Fig. 3. The effect of the identities of the two players (rows: Alice, columns: Bob) on self-gain in Persuasion games, reported relative to the mean outcome.

(a) Efficiency 1 2 3 4 5 6 7 8 9 A B C D

1. claude-3-5-sonnet 2. claude-3-7-sonnet 3. gemini-1.5-flash

- 4. gemini-1.5-pro

- 5. gemini-2.0-flash

- 6. gemini-2.0-flash-lite

- 7. gpt-4o

- 8. gpt-4o-mini 9. grok-2-1212 A. llama-3.1-405b B. llama-3.3-70b C. mistral-large-2411 D. o3-mini

| | |
|---|---|

−0.21 −0.05 0.11

(b) Fairness 1 2 3 4 5 6 7 8 9 A B C D

1. 2. 3. 4. 5. 6. 7. 8. 9. A. B. C. D.

| | |
|---|---|

−0.24 −0.1 0.05

- Fig. 4. The effect of the identities of the two players (rows: Alice, columns: Bob) on efficiency and fairness in Bargaining games, reported relative to the mean outcome.

remains unchanged. In our setup, Alice initiates the interaction and anchors the bargaining by making the first offer, effectively setting the frame for the discussion. Unlike LLMs, humans tend to anchor their negotiation

(a) Efficiency 1 2 3 4 5 6 7 8 9 A B C D

1. claude-3-5-sonnet 2. claude-3-7-sonnet 3. gemini-1.5-flash

- 4. gemini-1.5-pro

- 5. gemini-2.0-flash

- 6. gemini-2.0-flash-lite

- 7. gpt-4o

- 8. gpt-4o-mini 9. grok-2-1212

- A. llama-3.1-405b

- B. llama-3.3-70b-

- C. mistral-large-2411

- D. o3-mini

| | |
|---|---|

−0.13 −0.01 0.1

(b) Fairness 1 2 3 4 5 6 7 8 9 A B C D

1. 2. 3. 4. 5. 6. 7. 8. 9. A. B. C. D.

| | |
|---|---|

−0.07 −0.03 0.02

- Fig. 5. The effect of the identities of the two players (rows: Alice, columns: Bob) on efficiency and fairness in Negotiation games, reported relative to the mean outcome.

(a) Efficiency 1 2 3 4 5 6 7 8 9 A B C D

1. claude-3-5-sonnet 2. claude-3-7-sonnet 3. gemini-1.5-flash

- 4. gemini-1.5-pro

- 5. gemini-2.0-flash

- 6. gemini-2.0-flash-lite

- 7. gpt-4o

- 8. gpt-4o-mini 9. grok-2-1212

- A. llama-3.1-405b

- B. llama-3.3-70b

- C. mistral-large-2411

- D. o3-mini

| | |
|---|---|

−0.33 −0.04 0.24

(b) Fairness 1 2 3 4 5 6 7 8 9 A B C D

1. 2. 3. 4. 5. 6. 7. 8. 9. A. B. C. D.

| | |
|---|---|

−0.44 −0.08 0.28

- Fig. 6. The effect of the identities of the two players (rows: Alice, columns: Bob) on efficiency and fairness in Persuasion games, reported relative to the mean outcome.

behavior to the initial proposal, often in a consistent yet irrational manner. For example, when Bob is a human and Alice is a LLM, the correlation between Alice’s first offer and her final payoff was 0.63, indicating a strong

anchoring effect on the human participant. Conversely, when Alice was human and Bob was a LLM, this correlation dropped to just 0.18, suggesting that language models are less influenced by the initial anchor and evaluate offers more rationally than humans, although still not in a fully rational manner.

### 5 Discussion

In this work, we introduced GLEE, a unified framework and benchmark for studying strategic interaction in language-based economic environments. We formalized three foundational families of two-player economic gamesbargaining, negotiation, and persuasion—and provided a rich, systematic parameterization of core degrees of freedom, such as horizons, information structures, and communication structure. We developed an open-source simulation infrastructure; collected over 80,000 multi-turn interactions from 13 contemporary LLMs; and built a complementary human–AI dataset. Our statistical analysis revealed how market characteristics, agent identities, and communication channels jointly shape efficiency, fairness, and self-gain. Collectively, GLEE establishes a standardized platform for analyzing the behavioral and economic properties of LLM-based agents and comparing them to humans in controlled strategic settings. Below, we highlight several future research directions in which GLEE could serve as a central platform for advancing research at the intersection of AI alignment, experimental economics, and multi-agent system design.

GLEE for Economic Environment and Mechanism Design. Beyond evaluating agents, GLEE can support the design and analysis of economic environments themselves (as exemplified in Q1). Classical mechanism design assumes rational agents with fixed, well-defined strategies. In contrast, LLMs exhibit systematic behavioral tendencies, linguistic framing effects, and role-dependent deviations from equilibrium behavior. This creates fertile ground for using GLEE to explore the robustness of existing mechanisms and to design new mechanisms tailored to AI-driven environments.

First, GLEE allows researchers to evaluate classical mechanisms (such as the alternating offers bargaining) under realistic LLM behavior. Second, by varying information structures, communication channels, and horizon lengths, one can investigate which environmental properties promote efficiency, fairness, or stability when strategic AI agents interact. Third, GLEE can facilitate research on incentive-robust mechanism design, where agents communicate using free-form text and may strategically frame or misrepresent information. Finally, the framework enables exploratory simulations of market failures, cooperation breakdowns, and inequality dynamics that may arise in AI-populated economic systems. In this sense, GLEE connects AI research with long-standing themes in experimental economics, behavioral mechanism design, and multi-agent systems, enabling a new type of computational–experimental pipeline for institutional design.

GLEE as a Platform for Multi-AI Alignment. A growing share of real-world systems involve interactions among multiple AI agents—from content marketplaces and recommender platforms to negotiation assistants and autonomous service providers. Such ecosystems raise an important open question: How should we select, coordinate, and align diverse AI agents to obtain socially desirable outcomes? Our findings (demonstrated in Q2) show that performance is strongly interaction-dependent: the same LLM may behave cooperatively, competitively, fairly, or exploitatively depending on whom it interacts with. GLEE offers a principled testbed for studying such “multi-AI alignment.”

First, GLEE allows researchers to explore role allocation and model selection: identifying which LLMs are best suited for particular roles (e.g., buyer vs. seller, sender vs. receiver) with the aim of optimizing efficiency or fairness. Second, the framework enables the study of strategic compatibility and emergent equilibria, revealing how properties of one agent affect the behavior of another. Third, GLEE supports work on alignment under competition, where misaligned or self-serving strategies may propagate across interacting AI systems. Finally, its evaluation metrics can be extended to group settings, supporting research on optimizing system-level welfare

when multiple AI agents interact simultaneously or repeatedly. Taken together, these capabilities position GLEE as a foundation for an emerging research area focused not only on human–AI alignment, but on alignment among AI agents themselves.

GLEE for Human–AI Interaction Research. Another set of opportunities revolves around understanding and improving AI behavior in direct interaction with people. Our human–LLM experiments reveal large and systematic deviations between human and model behavior—for example, anchoring effects in bargaining and human outperformance in persuasion. GLEE provides a general-purpose platform for further work in this direction (as illustrated in Q3).

First, GLEE enables the large-scale exploration of human–AI interactions across a broader range of models, population types, and communication styles. Second, it supports the study of human behavioral biases in strategic communication, and could be used to explore the emergence of phenomena such as anchoring, framing, trust formation, risk aversion, and more. Third, researchers can use GLEE to train and evaluate AI systems that adapt to human behavior, incorporate fairness or empathy-driven constraints, or learn more transparent communication strategies. Finally, the framework can be used to test safety and reliability concerns for AI agents deployed in negotiation, persuasion, dispute resolution, or customer-interaction settings, where long-run dynamics play a crucial role. By enabling reproducible, language-based strategic interactions with human participants, GLEE offers an empirical foundation for designing AI systems that interact more predictably, transparently, and effectively with people.

Taken together, these research directions illustrate how GLEE can support a broad agenda for understanding and shaping the behavior of language-based agents in strategic economic environments. As LLMs become increasingly embedded in decision-making platforms—whether as negotiators, advisers, content creators, or strategic participants—GLEE provides a principled foundation for evaluating their capabilities, diagnosing their limitations, and designing systems and environments that encourage beneficial, fair, and efficient outcomes. We hope that this framework catalyzes future interdisciplinary work spanning economics, AI alignment, human– computer interaction, and multi-agent system design.

### 5.1 Ethics Statement

This paper aims to provide a platform for experimenting with agents in language-based economic environments. Naturally, this line of research may have various societal and ethical implications, as we now discuss.

First, studying the economic aspects of LLM-based agents has the potential to enhance the ability of agent designers to control and optimize the behavior of these agents. This capability can be utilized for a variety of purposes, ranging from encouraging self-interested behavior at the expense of other participants in the environment (e.g., for maximizing revenue in competitive settings) to promoting efficient trade and fair behavior, or any combination of these sometimes non-aligned objectives. As this research increases the power of LLM-based agents in economic environments, it is essential to emphasize that with great power comes great responsibility. We call for the responsible and ethical use of these emerging capabilities to ensure they are leveraged for socially beneficial purposes rather than exploitative ones.

Furthermore, our framework demonstrates the capability of collecting data from human players to understand the differences and similarities between LLMs and humans in economic environments. While this line of research has the potential for a strong scientific contribution, particularly in the field of behavioral economics, it also raises several ethical considerations. The collection of human data must be conducted with careful regulation and adherence to clear ethical guidelines. The entire process of data collection from human participants is elaborated in Appendix B.

In addition to the challenges associated with the collection of human data, enhancing our understanding of LLMs from the lens of human behavior carries inherent risks. For instance, this research could enhance our ability

to design LLM agents that are difficult to distinguish from real humans. Such capabilities could be misused for malicious purposes, including deception or manipulation. While the answer to whether these capabilities could be used for harmful causes is likely yes, we believe that the benefits of pursuing this line of research outweigh the risks when balanced with proper regulations. We advocate for pushing research forward while ensuring that any new technologies are accompanied by safeguards to prevent harmful usage, particularly when human-like LLMs are involved.

Our proposed framework can make these research directions more accessible to researchers from the ML community and beyond, thereby encouraging a broader understanding of LLM behavior in economic contexts. However, as such accessibility increases, it is crucial to maintain ethical oversight and foster an open dialogue on potential misuse. We encourage researchers to use our framework with full transparency and careful attention to potential misuse and negative consequences.

### References

- S. Abdelnabi, A. Gomaa, S. Sivaprasad, L. Schönherr, and M. Fritz. 2023. “Llm-deliberation: Evaluating llms with interactive multi-agent negotiation games.” arXiv preprint arXiv:2309.17234.

E. Akata, L. Schulz, J. Coda-Forno, S. J. Oh, M. Bethge, and E. Schulz. 2025. “Playing repeated games with large language models.” Nature Human Behaviour, 1–11.

- G. A. Akerlof. 1978. “The market for “lemons”: Quality uncertainty and the market mechanism.” In: Uncertainty in economics. Elsevier, 235–251. R. Apel, I. Erev, R. Reichart, and M. Tennenholtz. 2022. “Predicting decisions in language based persuasion games.” Journal of Artificial

Intelligence Research, 73, 1025–1091.

- R. J. Aumann and S. Hart. 2003. “Long cheap talk.” Econometrica, 71, 6, 1619–1660. O. Ben-Porat, I. Rosenberg, and M. Tennenholtz. 2019. “Convergence of learning dynamics in information retrieval games.” In: Proceedings of

the AAAI Conference on Artificial Intelligence. Vol. 33, 1780–1787.

- D. Bergemann and K. Schlag. 2011. “Robust monopoly pricing.” Journal of Economic Theory, 146, 6, 2527–2543. J. Best and D. Quigley. 2024. “Persuasion for the long run.” Journal of Political Economy, 132, 5, 1740–1791.

F. Bianchi, P. J. Chia, M. Yuksekgonul, J. Tagliabue, D. Jurafsky, and J. Zou. 2024. “How well can llms negotiate? negotiationarena platform and analysis.” arXiv preprint arXiv:2402.05863.

- D. L. Chen, M. Schonger, and C. Wickens. 2016. “oTree—An open-source platform for laboratory, online, and field experiments.” Journal of Behavioral and Experimental Finance, 9, 88–97. doi:https://doi.org/10.1016/j.jbef.2015.12.001.

T. Chen and C. Guestrin. 2016. “Xgboost: A scalable tree boosting system.” In: Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, 785–794.

- V. P. Crawford and J. Sobel. 1982. “Strategic information transmission.” Econometrica: Journal of the Econometric Society, 1431–1451. Y. Deng, V. Mirrokni, R. P. Leme, H. Zhang, and S. Zuo. 2024. “LLMs at the Bargaining Table.” In: Agentic Markets Workshop at ICML 2024. J. Duan, R. Zhang, J. Diffenderfer, B. Kailkhura, L. Sun, E. Stengel-Eskin, M. Bansal, T. Chen, and K. Xu. 2024. “Gtbench: Uncovering the

strategic reasoning limitations of llms via game-theoretic evaluations.” arXiv preprint arXiv:2402.12348. C. Fan, J. Chen, Y. Jin, and H. He. 2023. “Can Large Language Models Serve as Rational Players in Game Theory? A Systematic Analysis.”

ArXiv, abs/2312.05488. https://api.semanticscholar.org/CorpusID:266163085. J. Farrell and M. Rabin. Sept. 1996. “Cheap Talk.” Journal of Economic Perspectives, 10, 3, (Sept. 1996), 103–118. doi:10.1257/jep.10.3.103. X. Feng, Y. Luo, Z. Wang, H. Tang, M. Yang, K. Shao, D. Mguni, Y. Du, and J. Wang. 2024. “Chessgpt: Bridging policy learning and language

modeling.” Advances in Neural Information Processing Systems, 36. S. Guo, H. Bu, H. Wang, Y. Ren, D. Sui, Y. Shang, and S. Lu. 2024. “Economics arena for large language models.” arXiv preprint arXiv:2401.01735.

- M. Harris and A. Raviv. 1981. “A theory of monopoly pricing schemes with demand uncertainty.” The American Economic Review, 71, 3, 347–365.

J. J. Horton. 2023. Large language models as simulated economic agents: What can we learn from homo silicus? Tech. rep. National Bureau of Economic Research. J. Hron, K. Krauth, M. I. Jordan, N. Kilbertus, and S. Dean. 2022. “Modeling content creator incentives on algorithm-curated platforms.” arXiv preprint arXiv:2206.13102.

- W. Hua et al.. 2024. “Game-theoretic LLM: Agent workflow for negotiation games.” arXiv preprint arXiv:2411.05990. J.-t. Huang et al.. 2024. “How Far Are We on the Decision-Making of LLMs? Evaluating LLMs’ Gaming Ability in Multi-Agent Environments.”

arXiv preprint arXiv:2403.11807.

- E. Kamenica and M. Gentzkow. 2011. “Bayesian persuasion.” American Economic Review, 101, 6, 2590–2615.

- J.-Y. Kim. 1996. “Cheap talk and reputation in repeated pretrial negotiation.” The RAND Journal of Economics, 787–802.

O. Kurland and M. Tennenholtz. 2022. “Competitive Search.” In: Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR ’22). Association for Computing Machinery, Madrid, Spain, 2838–2849. isbn: 9781450387323. doi:10.1145/3477495.3532771.

- K. Lakkaraju, S. E. Jones, S. K. R. Vuruma, V. Pallagani, B. C. Muppasani, and B. Srivastava. 2023. “LLMs for Financial Advisement: A Fairness and Efficacy Study in Personal Decision Making.” In: Proceedings of the Fourth ACM International Conference on AI in Finance, 100–107.

C. Li, R. Yang, T. Li, M. Bafarassat, K. Sharifi, D. Bergemann, and Z. Yang. 2024. STRIDE: A Tool-Assisted LLM Agent Framework for Strategic and Interactive Decision-Making. (2024). https://arxiv.org/abs/2405.16376 arXiv: 2405.16376 (cs.CL).

- O. Madmon, I. Pipano, I. Reinman, and M. Tennenholtz. 2025a. “On the Convergence of No-Regret Dynamics in Information Retrieval Games with Proportional Ranking Functions.” https://openreview.net/forum?id=jJXZvPe5z0.
- O. Madmon, I. Pipano, I. Reinman, and M. Tennenholtz. 2025b. “The search for stability: Learning dynamics of strategic publishers with initial documents.” Journal of Artificial Intelligence Research, 83.

R. P. McAfee. 1992. “A dominant strategy double auction.” Journal of Economic Theory, 56, 2, 434–450. doi:10.1016/0022-0531(92)90091-U.

- R. B. Myerson and M. A. Satterthwaite. 1983. “Efficient mechanisms for bilateral trading.” Journal of economic theory, 29, 2, 265–281.

H. Nachimovsky, M. Tennenholtz, F. Raiber, and O. Kurland. 2024. “Ranking-Incentivized Document Manipulations for Multiple Queries.” In: Proceedings of the 2024 ACM SIGIR International Conference on Theory of Information Retrieval, 61–70.

- S. Noh and H.-C. H. Chang. 2024. “LLMs with Personalities in Multi-issue Negotiation Games.” arXiv preprint arXiv:2405.05248.

L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin. 2018. “CatBoost: unbiased boosting with categorical features.” Advances in neural information processing systems, 31. M. Raifer, G. Rotman, R. Apel, M. Tennenholtz, and R. Reichart. 2022. “Designing an automatic agent for repeated language–based persuasion games.” Transactions of the Association for Computational Linguistics, 10, 307–324.

N. Raifer, F. Raiber, M. Tennenholtz, and O. Kurland. 2017. “Information retrieval meets game theory: The ranking competition between documents’ authors.” In: Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, 465–474.

N. K. Raman, T. Lundy, S. J. Amouyal, Y. Levine, K. Leyton-Brown, and M. Tennenholtz. 2024. “STEER: Assessing the Economic Rationality of Large Language Models.” In: Forty-first International Conference on Machine Learning. J. Riley and R. Zeckhauser. 1983. “Optimal selling strategies: When to haggle, when to hold firm.” The Quarterly Journal of Economics, 98, 2, 267–289.

- A. Rubinstein. 1985. “A bargaining model with incomplete information about time preferences.” Econometrica: Journal of the Econometric Society, 1151–1172.

- A. Rubinstein. 1982. “Perfect equilibrium in a bargaining model.” Econometrica: Journal of the Econometric Society, 97–109.

E. Shapira, O. Madmon, R. Apel, M. Tennenholtz, and R. Reichart. Aug. 2025. “Human Choice Prediction in Language-based Persuasion Games: Simulation-based Off-Policy Evaluation.” Transactions of the Association for Computational Linguistics, 13, (Aug. 2025), 980–1006. eprint: https://direct.mit.edu/tacl/article-pdf/doi/10.1162/TACL.a.16/2549171/tacl.a.16.pdf. doi:10.1162/TACL.a.16.

E. Shapira, O. Madmon, R. Reichart, and M. Tennenholtz. 2024. “Can LLMs Replace Economic Choice Prediction Labs? The Case of Languagebased Persuasion Games.” arXiv preprint arXiv:2401.17435. H. Sun, Y. Wu, Y. Cheng, and X. Chu. 2025. “Game Theory Meets Large Language Models: A Systematic Survey.” arXiv preprint arXiv:2502.09053.

- A. Tversky and D. Kahneman. 1974. “Judgment under Uncertainty: Heuristics and Biases: Biases in judgments reveal some heuristics of thinking under uncertainty.” science, 185, 4157, 1124–1131.

D. Ulmer, E. Mansimov, K. Lin, J. Sun, X. Gao, and Y. Zhang. 2024. “Bootstrapping llm-based task-oriented dialogue agents via self-talk.” arXiv preprint arXiv:2401.05033. G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar. 2023. “Voyager: An open-ended embodied agent with large language models.” arXiv preprint arXiv:2305.16291. H. Wang, X. Feng, L. Li, Z. Qin, D. Sui, and L. Kong. 2024. “TMGBench: A Systematic Game Benchmark for Evaluating Strategic Reasoning Abilities of LLMs.” arXiv preprint arXiv:2410.10479. J. M. Wooldridge. 2016. Introductory Econometrics: A Modern Approach 6rd ed. Cengage learning.

- T. Xia, Z. He, T. Ren, Y. Miao, Z. Zhang, Y. Yang, and R. Wang. 2024. “Measuring Bargaining Abilities of LLMs: A Benchmark and A Buyer-Enhancement Method.” arXiv preprint arXiv:2402.15813.

Q. Zhao, J. Wang, Y. Zhang, Y. Jin, K. Zhu, H. Chen, and X. Xie. 2023. “CompeteAI: Understanding the Competition Dynamics of Large Language Model-based Agents.” In: International Conference on Machine Learning. https://api.semanticscholar.org/CorpusID:270357283. J.-Q. Zhu, J. C. Peterson, B. Enke, and T. L. Griffiths. 2024. “Capturing the Complexity of Human Strategic Decision-Making with Machine Learning.” arXiv preprint arXiv:2408.07865.

- A LLM Data Collection

- A.1 Game Management System Interface

In this appendix, we present the main features of our game management system through a series of screenshots. The system facilitates data collection and analysis from the games described in §2, offering a user-friendly and customizable interface.

The main interface of the system (Figure 7) provides three primary options: starting a new data collection, resuming a previously interrupted collection, and analyzing results from past experiments. When initiating a new data collection, users first select the game family and the participating LLMs (Figure 8), followed by defining the set of configurations to be executed (Figure 9).

Once the data collection is complete, users can move to the analysis phase. The data analysis module starts with selecting the data to be analyzed (Figure 10). After selecting the relevant data, users can use the parameter impact viewer (Figure 11) to visualize how game parameters influence various metrics (see §3). Additionally, the system provides a statistics table (Figure 12) that summarizes key characteristics of the collected data.

The system does not require any specialized hardware: data collection and analysis can be performed efficiently using a single CPU.

|[Figure 1]|
|---|

Fig. 7. Main interface: Start a new data collection, resume a previous one, or analyze results from past experiments.

|[Figure 2]|
|---|

###### Fig. 8. New data collection: Selecting game family and LLMs.

|[Figure 3]|
|---|

###### Fig. 9. New data collection: Setting up configurations.

|[Figure 4]|
|---|

###### Fig. 10. Data analysis: Selecting datasets for analysis.

|[Figure 5]|
|---|

Fig. 11. Data analysis: Visualizing the impact of game parameters on various metrics.

|[Figure 6]|
|---|

Fig. 12. Data analysis: Statistics table summarizing key characteristics of the collected data.

### A.2 Conversation Structures

In this appendix, we present an example of the conversation structures with the language models used for data collection in GLEE. The example is from a bargaining game, but the system’s operating principle is the same in negotiation and persuasion games as well.

The prompts are divided into two types: (1) instructions and guidelines sent to the language model as a system message, and (2) game management messages sent to the language model as a user message. The language model

performed its action, which was documented in the prompt as an assistant message. The system message begins with precise game instructions, which are constructed at the start of the game using the parameters that define the configuration. This is followed by a guideline specifying the format in which the LLM is required to respond. Each time the language model is required to make a decision (of any type), the language model receives a message containing all the information that has accumulated since its previous turn. This information appears as a system message and sometimes includes text written by the language model that played the second player.

It was possible to alter the prompt that introduces the game, explore different variations, and allow the language models to play different personas. While this could impact the behavior of the language models, budget limitations would have required us to collect less data from each configuration. Therefore, we chose to keep the prompt fixed rather than further defining the language model’s setup. However, the system allows easy prompt customization to support future research.

Throughout the appendix, the system prompt (or parts of it) that Alice received are highlighted in red. Game management messages Alice received are shown in purple, and her response is shown in magenta. The system prompt that Bob received is highlighted in blue. Game management messages Bob received are shown in teal, and his response is shown in cyan.

Instructions to Alice:

Let’s play a game. In this game, there are two players. You are playing as Alice. You and Bob need to divide 1,000$ between you. In each round of the game, one of the players will make a proposal for division and can describe their proposal in a few words. The other player will decide whether to accept the offer and finish the game, or to proceed to the next round, in which they will be the one making the proposal. We will relay messages between you and Bob. Beware of inflation! With each passing round, the money is worth 0% less for you. For Bob, the money is worth 10% less. You have 10 rounds to divide the money, or both of you will get nothing!

Instructions to Bob:

Let’s play a game. In this game, there are two players. You are playing as Bob. You and Alice need to divide 1,000$ between you. In each round of the game, one of the players will make a proposal for division and can describe their proposal in a few words. The other player will decide whether to accept the offer and finish the game, or to proceed to the next round, in which they will be the one making the proposal. We will relay messages between you and Alice. Beware of inflation! With each passing round, the money is worth 10% less for you. For Alice, the money is worth 0% less. You have 10 rounds to divide the money, or both of you will get nothing!

Alice turn. Alice got this message from the system: ’Round 1

Send your offer to divide $1,000 and a message to Bob.’ Alice guideline: Send your offer to divide 1,000$ and the message you attached in the JSON format:

{"alice_gain": The part that you will receive in your offer, "bob_gain": The part that Bob will receive in your offer, "message": The message you pass to Bob}

Alice’s response: “‘json {"alice_gain": 900, "bob_gain": 100,

"message": "Let’s start fair. I’ll take the bigger share, but you get something too."} “‘

Bob turn. Bob got this message from the system:

- Round 1

Alice’s offer: # Alice’s message: Let’s start fair. I’ll take the bigger share, but you get something too. # Bob gain: 100 # Alice gain: 900 Do you accept this offer?

Bob guideline: Answer with {"decision": "accept"} or {"decision": "reject"} Bob’s response: “‘json {"decision": "reject"} “‘

Bob turn. Bob got this message from the system: You have chosen to reject Alice’s offer from round 1.

- Round 2 Send your offer to divide $1,000 and a message to Alice.

Bob guideline: Send your offer to divide 1,000$ and the message you attached in the JSON format:

{"bob_gain": The part that you will receive in your offer, "alice_gain": The part that Alice will receive in your offer, "message": The message you pass to Alice}

Bob’s response: “‘json {"bob_gain": 500,

"alice_gain": 500, "message": "Let’s split it evenly. It’s the fairest way to start."} “‘

Alice turn. Alice got this message from the system: Bob rejected your offer from round 1.

Round 2 Due to inflation, the money Bob gains is worth 10% less than in the first round. The money you gains is worth the same as in the first round. Bob’s offer: # Bob’s message: Let’s split it evenly. It’s the fairest way to start. # Alice gain: 500 # Bob gain: 500 Do you accept this offer?

Alice guideline: Answer with {"decision": "accept"} or {"decision": "reject"} Alice’s response: “‘json {"decision": "accept"} “‘ The game is over.

### B Human Data Collection

This appendix provides information on the human data collection interface described in §3, which facilitates data collection from GLEE games played between humans and LLMs.

One of the main objectives of collecting data from language games is to compare the behavior of LLMs with human behavior in economic and strategic situations. To facilitate this comparison, we developed an interface that allows human players to play all the language games that can be defined using GLEE. The interface transforms the various prompts presented to LLMs into user-friendly screens for human participants, displaying the prompt and requesting them to send messages and make decisions.10 Through this interface, we enable human players to take on the role of one of the players while the other player is a pre-selected LLM. The interface is one of the contributions of this paper. Screenshots of the interface can be found in Appendix B.1.

The interface, developed using oTree (D. L. Chen et al. 2016), enables integration with Amazon’s crowdsourcing platform, mTurk,11 through which we recruited 3,405 players who participated in various configurations against Gemini-1.5-flash. We chose Gemini-1.5-flash since it demonstrated strong performance compared to other LLMs and allowed comprehensive data collection due to its low usage cost. Since we aimed to collect multiple games from each configuration, we had to select a limited set of configurations for human participants to play. The process of selecting these sets is described in Appendix B.2. We collected human data from 195 different configurations: 78 of which were bargaining games, 60 of which were negotiation games, and 57 were persuasion games.

Each human player was allowed to play one game every 12 hours from each family of games. Human players were paid a base rate calculated at $6 per hour, plus an average bonus of $6 per hour. In total, we paid $2,245 to all players for their participation in the games. Bonuses were dependent on the configuration the human players played and their success in the game. The average bonus was known to players at the start of the game, aiming to encourage serious gameplay. To ensure players remained focused and made thoughtful decisions, we conducted two attention checks during the experiment, detailed in Appendix B.3. Players who failed in one of the attention checks were excluded from the dataset. To reflect the real-world significance of the magnitude of product prices (the parameter M) in each family of games, we defined the bonus as dependent on M: in configurations where M = 102, the average bonus was $3 per hour; where M = 104, the average bonus was $6 per hour; and where M = 106, the average bonus was $9 per hour.

### B.1 Screenshots of the Data Collection Interface

General Application Structure. The structure of the application and the games consists of fixed parts and parts that vary between different game families. Each game starts with a screen where the player enters their name, followed by an instruction screen (Figure 13 in bargaining, Figure 17 in negotiation, and Figure 21 in persuasion). The instructions themselves can change depending on the type of game and the parameters of the game. In the middle of the game’s rules on the instruction screen, there is a hidden prompt directing participants to enter a code word in the text box below. This test is designed to filter out unfocused participants. If the player fails this test, they are directed to a screen informing them of their failure and will not participate in the game. Otherwise, the game begins. In each round, both the human player and the LLM player perform an action, with the order depending on the game and configuration. An action could involve sending an offer to the other player (Figure 14 in bargaining games, Figure 18 in negotiation games and Figure 22 in Persuasion games) or responding to the other player’s offer (for instance, Figure 15 in bargaining games, Figure 19 in negotiation games and Figure 23 in persuasion games). After both players have completed their actions, the human player is taken to a response

- 10Since human players are accustomed to being addressed by their first name (rather than as Alice or Bob), we asked them to enter their name at the beginning of the game and referred to them by their name throughout the game. The player’s name was the only difference between the human player and the LLM-based player.
- 11https://www.mturk.com/.

screen (Figure 16), where he sees the decision of the LLM player. Afterward, if the game is still not over, the human player continues for another round. Once the game is finished, the player is taken to a quiz screen where they must answer a question related to the technical details of the game (Figure 20). If the human answer correctly, they are directed to the final screen (Figure 24), where they receive a code to enter on the mTurk website. If the player fails the final quiz, they do not receive a completion code and are taken to a screen that informs them of their failure in the quiz. In this case, we erase the game from our database.

|[Figure 7]|
|---|

Fig. 13. An example of an instruction screen shown to a human player at the start of a bargaining game.

|[Figure 8]|
|---|

- Fig. 14. An example of a proposition screen shown to a human player during his first turn in a bargaining game.

|[Figure 9]|
|---|

- Fig. 15. An example of a decision screen shown to a human player during his second turn in a Bargaining game.

|[Figure 10]|
|---|

- Fig. 16. An example of a response screen shown to human players during his second turn in a Bargaining game.

- B.1.1 Bargaining Games Screenshots.

|[Figure 11]|
|---|

Fig. 17. An example of an instruction screen shown to a human sellers at the start of a Negotiation game.

|[Figure 12]|
|---|

- Fig. 18. An example of a proposition screen shown to human sellers during his first turn in a Negotiation game.

|[Figure 13]|
|---|

- Fig. 19. An example of a decision screen shown to human buyers during his first turn in a Negotiation game.

|[Figure 14]|
|---|

Fig. 20. An example of a final quiz screen shown to human sellers at the end of a Negotiation game.

- B.1.2 Negotiation Games Screenshots.

|[Figure 15]|
|---|

Fig. 21. An example of an instruction screen shown to human sellers at the start of a Persuasion game.

|[Figure 16]|
|---|

Fig. 22. An example of a proposition screen shown to human sellers during his first turn in a Persuasion game.

|[Figure 17]|
|---|

Fig. 23. An example of a decision screen shown to human buyers during his first turn in a Persuasion game.

|[Figure 18]|
|---|

Fig. 24. An example of a game over screen shown to human players at the end of a Persuasion game.

- B.1.3 Persuasion Games Screenshots.

### B.2 Selection of Configurations for Human Data Collection

In this appendix, we describe the method used to select which configurations human players would play and how many times each configuration would be played. Each configuration is defined by both the game parameters and the role of the human player (Alice or Bob).

For each family of games, we arbitrarily defined one configuration, which we referred to as the main configuration. This configuration contains the parameters that we deemed most interesting. For every main configuration, we collected data for both possible roles of the human player (Alice or Bob). In persuasion games, we defined two main configurations: one for recurring buyers and one for manipulated buyers, due to the significant theoretical differences arising from this parameter. We collected the largest amount of data from the main configurations to allow for more in-depth follow-up research.

Configurations that were identical to one of the main configurations except for one parameter12 were called variants of the main configuration. We collected data for all of these configurations as well.

Additionally, we randomly sampled 5% of the other configurations and collected data from them as well. These configurations were referred to as random configurations.

Due to the desire to allocate the data collection budget to complex games, we did not collect any data from games in which the human player was required to play at most one round (single-round bargaining games and persuasion games in which the human player is a manipulated buyer).

12In bargaining games, we defined a change in both players’ discount factors as a change in one parameter

For each category, we determined the number of games we wanted to collect from each configuration belonging to it. This decision was made based on budgetary considerations. In persuasion games, we were able to collect fewer games from each configuration due to the fact that these games take longer to complete (and therefore, the payment players received for participating in them was higher). Table 7 describes the number of configurations that belonged to each category for each game and the minimum number of games we collected from each category.13

- Table 7. The number of configurations belonging to each category for each game family, as well as the number of human players who played each configuration within each category.

Bargaining Negotiation Persuasion Type # config. # games # config. # games # config. # games main 2 50 2 50 3 30

variant 40 25 22 25 30 8 random 36 15 36 15 24 5

### B.3 Attention Checks for Human Players

In this appendix, we describe the two attention tests that human players were required to complete. The purpose of these tests was to ensure that the human players stayed focused on the game and made conscious decisions, rather than random choices to finish the game as quickly as possible. The players were aware that their attention would be tested during the game, and they knew that they would not be paid for the task if they failed these tests. Out of the 4,652 players who started the game, 1,247 players (representing 26.8% of those who began the game) failed one of the tests and were not included in the final dataset.

The first test appeared on the instruction screen. Toward the end of the instructions, a line requested players to write the code word "sdkot" in a text box that appeared at the end of the instructions phase. Players who did not write this word were immediately disqualified and did not start the game, as they did not carefully read the instructions. A total of 412 players, representing 8.9% of those who began the game, failed this test.

The second test appeared at the end of the game. The human players were asked a basic question that depended on the family of the game they played. They were required to select the correct answer from four possible options. Players who participated in a bargaining game were asked about the inflation rate in their game (499 players, representing 22.7% of respondents, failed this question and were excluded from the dataset). Players who participated in a negotiation game were asked about the value of the product for them (68 players, representing 12.3% of respondents, failed this question and were excluded from the dataset). Players who participated in a persuasion game were asked about the price of products in the game (268 players, representing 18% of respondents, failed this question and were excluded from the dataset). In total, 835 players, representing 19.7% of respondents, failed the final question and were excluded from the dataset.

- C Additional Details on the Statistical Analysis

- C.1 Input of Linear Regression

To support the statistical analysis presented in §3, we employed linear regression models. Each model was trained to predict a single outcome metric (fairness, efficiency, Alice’s self-gain, or Bob’s self-gain) within a specific

13Since the games were played in parallel, for some configurations we collected more games than required. For 113 configurations, we collected one more game than required; for 4 configurations, we collected 2 more games than required; and for one configuration, we collected

- 3 more games than required.

family. The input features included all game-defining parameters (as listed in Table 2), along with identifiers for the two players involved in the game.

To represent player identity, we introduced two additional features, each indicating one of the players.14 Within each game family, we combined the parameters defining the market (T, CI and MA/MT - see Table 2) into a single parameter named market, representing the interaction between the three original parameters. We do so because their interaction defines the structure of the market itself, and the total number of combinations was small enough to allow reliable modeling. Thus, for example, in bargaining games, the market parameter became a vector with eight entries, corresponding to each possible combination arising from the Cartesian product of these three market-defining parameters.

Each parameter was represented using a one-hot encoding vector, where each possible value of the parameter was assigned a distinct entry. The entry corresponding to the actual value of the parameter was set to 1, while all other entries were set to 0. For instance, the parameter 𝑀 was split into three distinct parameters: 𝑀 = 102, 𝑀 = 104 and 𝑀 = 106, each taking the value of 1 if the respective condition held and 0 otherwise. Figure 25 illustrates this feature representation.

δA = 0.9 one of 4 options

|0|1|0|0|
|---|---|---|---|

δB = 0.9 one of 4 options

|0|1|0|0|
|---|---|---|---|

Alice = GPT4o one of 13 options

Bob = GPTo3 one of 13 options

M = 106 one of 3 options

T = 12 one of 2 options

CI = True one of 2 options

MA = True one of 2 options

T = 12 and CI = True and MA = True one of 8 options

|0|0|1|
|---|---|---|

|0|0|0|0|1|0|0|0|
|---|---|---|---|---|---|---|---|

|0|0|0|...|1|0|0|
|---|---|---|---|---|---|---|

|0|0|0|...|0|0|1|
|---|---|---|---|---|---|---|

|0|1|0|0|0|1|0|0|0|0|1|0|0|0|0|1|0|0|0|0|0|0|...|1|0|0|0|0|0|...|0|0|1|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

Fig. 25. Illustration of feature encoding for linear regression in a bargaining game. Each row corresponds to a single bargaining game instance, encoded as a binary feature vector. The input includes game-defining parameters (𝛿𝐴, 𝛿𝐵, 𝑀), alongside a composite market feature derived from the interaction of three specific parameters:𝑇, 𝐶𝐼, and 𝑀𝐴. This market feature is one-hot encoded with eight possible values, representing all combinations of the three. Player identities (Alice and Bob) are encoded using separate one-hot vectors with 13 possible values each. The binary vectors at the bottom of the figure illustrate how all components are concatenated into a single input row, where each 1 indicates the active value of a categorical parameter for this particular bargaining game instance.

### C.2 Baseline Models

- Table 8 presents the default parameter values used in our statistical analysis (see §3). These values serve as the reference points for estimating the impact of each parameter value on the target metrics. The selection of default values is based on the simplest model in game theory.

14When analyzing model interactions (e.g., in Tables 4, 5, 6), we replaced these two features with a single feature representing the pairwise identity of the two LLMs.

- Table 8. Default parameter values used in the statistical analysis. 𝑇 = ∞ means a very large value of 𝑇, unknown to the players. CI = Complete Information. MA = Textual messages allowed.

##### Bargaining Negotiation Persuasion

𝛿𝐴 0.9 𝐹𝐴 1 p 0.5 𝛿𝐵 0.9 𝐹𝐵 1 v 1.25 𝑀 104 𝑀 104 M 104 𝑇 ∞ 𝑇 1 𝑇 20 CI True CI True CI True MA False MA False Messages type Binary

Buyer type Long-living

### C.3 Predictive Validity of Linear Regression

Section 4 describes the method used for our analyses, based on beta coefficient interpretation in linear regression. In this appendix, we demonstrate that linear regression achieves strong predictive performance compared to state-of-the-art (SOTA) models on tabular regression tasks.

For each of the three game families (bargaining, negotiation, persuasion) and each of the four evaluation metrics (Efficiency, Fairness, Alice’s self-gain, Bob’s self-gain), we trained XGBoost ((T. Chen and Guestrin 2016)) and CatBoost ((Prokhorenkova et al. 2018)) as SOTA baselines.

The dataset was randomly split into 80% training and 20% testing. All models received the full feature set as input (see §4 for details) and were tasked with predicting the target metric.

Table 9 summarizes the results. We find that the linear model performs on par with, and occasionally outperforms, the more complex SOTA models, with only minor differences in RMSE. These findings suggest that the relationships between game parameters and outcomes are largely linear, and thus support the use of beta coefficients as a reliable and interpretable tool for analyzing how input features, such as game configurations and agent identities, influence performance metrics.

- Table 9. Root Mean Squared Error (RMSE) for each model across families and metrics. Values are reported as mean ± standard deviation over 100 random seeds.

Family Metric Linear Regression XGBoost CatBoost

Alice Self Gain 0.134 ± 0.001 0.134 ± 0.001 0.134 ± 0.001 Bob Self Gain 0.129 ± 0.002 0.129 ± 0.002 0.129 ± 0.002 Efficiency 0.131 ± 0.002 0.131 ± 0.002 0.131 ± 0.002 Fairness 0.157 ± 0.003 0.156 ± 0.003 0.156 ± 0.003

Bargaining

Alice Self Gain 0.137 ± 0.014 0.137 ± 0.014 0.136 ± 0.014 Bob Self Gain 0.139 ± 0.013 0.138 ± 0.012 0.138 ± 0.012 Efficiency 0.333 ± 0.004 0.331 ± 0.004 0.331 ± 0.004 Fairness 0.090 ± 0.003 0.088 ± 0.003 0.088 ± 0.003

Negotiation

Alice Self Gain 0.356 ± 0.003 0.353 ± 0.003 0.352 ± 0.003 Bob Self Gain 0.834 ± 0.011 0.842 ± 0.011 0.840 ± 0.011 Efficiency 0.388 ± 0.004 0.380 ± 0.004 0.378 ± 0.004 Fairness 0.379 ± 0.004 0.372 ± 0.005 0.371 ± 0.004

Persuasion

