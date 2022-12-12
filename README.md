# CSE 490G1 Final Project: Text Summarization

## Abstract

## Problem Statement

People's declining attention spans and willingness to read make it crucial to create concise summaries of significant news stories. To do this effectively, a good summarizer must not only select relevant words from the original text, but also create new grammatically correct phrases or sentences that accurately convey the main ideas of the article. A good summarizer can significantly reduce screening and reading time during research, ensuring most of the time is allocated for quality and relevant contents.

There are many different methods for summarizing text including extractive summarization and abstractive summarization.

- Extractive summarization involves selecting key phrases and sentences from the original text to create a summary. This method is based on the idea that the most important information in a piece of text can be found in its individual sentences and phrases. Some examples include sentence extraction, keyword extraction, Latent semantic analysis (with SVD), and TextRank.

- Abstractive summarization, on the other hand, uses deep learning algorithms to understand the meaning and context of the text, and then generate a summary that is written in the model's own words. It is more challenging as it requires the model to have a deep understanding of the text and be able to generate new text that accurately captures the main ideas of the original. Some examples models such as recurrent neural networks (RNNs) and transformers have gained a lot of attention in recent years.

In this project, I will explore many different approaches to summarize a given texts.

## Related Work

Today, there are still a lot of active works and researches on text summarization. Most of the work are focused on improving the state-of-the-art sequence-to-sequence (seq2seq) attention-based encoder-decoder architecture. How it works is summarized below:

- The encoder processes the input text and encodes it into a compact representation, called the context vector. This vector captures the most important information from the input text, such as the overall meaning and the relationships between different words or phrases.

- The decoder predicts the next word in the summary one at a time, based on the previous words in the summary and the context vector. It also uses an attention mechanism to focus on different parts of the input text at different times, allowing it to generate summaries that accurately capture the main ideas of the original text.

In this project, I implemented and trained a basic encoder-decoder model, with some additional preprocessing and optimization steps to improve my model. I use a Kaggle dataset [News Summary](https://www.kaggle.com/datasets/sunnysai12345/news-summary), which contains more than 4500 published articles from February to August 2017, with annotated headlines. I choose this dataset as its article length is shorter, hence reduces the complexity of training and also decreases of the dataset. The other dataset that I tried first took a lot of unnecessary Colab time to just download/upload the dataset.

Take a peek at an example of the dataset:

Article: Saurav Kant, an alumnus of upGrad and IIIT-B's PG Program in Machine learning and Artificial Intelligence, was a Sr Systems Engineer at Infosys with almost 5 years of work experience. The program and upGrad's 360-degree career support helped him transition to a Data Scientist at Tech Mahindra with 90% salary hike. upGrad's Online Power Learning has powered 3 lakh+ careers.

Headline: upGrad learner switches to career in ML & Al with 90% salary hike

## Methodology

### TextRank

First, in order to improve my understanding on text summarization, I started with a non-deep-learning model - TextRank. TextRank is an algorithm for summarizing text that is based on the principles of graph-based ranking algorithm PageRank. TextRank constructs a graph of the text, with each sentence represented as a vertex in the graph. The edges between the vertices are determined by the similarity between the sentences, with sentences that are more similar having stronger edges. The TextRank algorithm then applies the PageRank algorithm to this graph to score each sentence. The sentences with the highest scores are selected as the summary of the text, with the number of sentences in the summary determined by the desired length of the summary. In my work, TextRank has been shown to produce high-quality summaries of text.

For example:

Input paragraphs:
In an attempt to build an AI-ready workforce, Microsoft announced Intelligent Cloud Hub which has been launched to empower the next generation of students with AI-ready skills. Envisioned as a three-year collaborative program, Intelligent Cloud Hub will support around 100 institutions with AI infrastructure, course content and curriculum, developer support, development tools and give students access to cloud and AI services. As part of the program, the Redmond giant which wants to expand its reach and is planning to build a strong developer ecosystem in India with the program will set up the core AI infrastructure and IoT Hub for the selected campuses. The company will provide AI development tools and Azure AI services such as Microsoft Cognitive Services, Bot Services and Azure Machine Learning.According to Manish Prakash, Country General Manager-PS, Health and Education, Microsoft India, said, "With AI being the defining technology of our time, it is transforming lives and industry and the jobs of tomorrow will require a different skillset. This will require more collaborations and training and working with AI. That’s why it has become more critical than ever for educational institutions to integrate new cloud and AI technologies. The program is an attempt to ramp up the institutional set-up and build capabilities among the educators to educate the workforce of tomorrow." The program aims to build up the cognitive skills and in-depth understanding of developing intelligent cloud connected solutions for applications across industry. Earlier in April this year, the company announced Microsoft Professional Program In AI as a learning track open to the public. The program was developed to provide job ready skills to programmers who wanted to hone their skills in AI and data science with a series of online courses which featured hands-on labs and expert instructors as well. This program also included developer-focused AI school that provided a bunch of assets to help build AI skills.

Output:

Envisioned as a three-year collaborative program, Intelligent Cloud Hub will support around 100 institutions with AI infrastructure, course content and curriculum, developer support, development tools and give students access to cloud and AI services. The company will provide AI development tools and Azure AI services such as Microsoft Cognitive Services, Bot Services and Azure Machine Learning. According to Manish Prakash, Country General Manager-PS, Health and Education, Microsoft India, said, "With AI being the defining technology of our time, it is transforming lives and industry and the jobs of tomorrow will require a different skillset

However, the limitation of extractive summarization makes it hard to condense an article into a short news headline. It is more useful to create a shorter paragraph that summarize a long article.

Hence, I start to build an encoder-decoder model in Pytorch.

### Encoder-Decoder (GRU)

#### Preprocess

First, I use the following techniques to preprocess the data

1. Stopwords are common words that do not have much meaning on their own, such as "the," "a," and "and." These words are removed as they do not contribute much to the meaning of the text and can hinder the performance of the model.

2. Contraction mapping involves replacing contractions, such as "can't" and "won't," with their expanded forms, such as "cannot" and "will not." This can help improve the performance of the NLP model because contractions are often treated as separate words by the model, which can lead to incorrect predictions.

3. Removing punctuation involves removing punctuation marks from the text, such as commas, periods, and exclamation points. This can help the NLP model focus on the words in the text and their relationships, rather than being distracted by punctuation marks that do not contribute much to the meaning of the text.

4. Make all characters lowercase. This can make it easier to identify the unique words and count how many times each one appears despite their different capitalization in the document, even though it also creates some other problem (Is 'US' and 'us' the same?)

5. Replace all occurence of '. ' to ' . ' ensure that tokenization removes the EOS from its prepended ending words.

#### Vocabulary

Initially, I tried to implement the Dataset and Dataloader class but I couldnt't fix a diemnsionality error that keeps bugging me. Hence, I keep the data in memory as a list instead. This indeed makes the training and batch organization a bit harder, but I managed to translate them accordingly.

#### Model

This model involves two parts: RNNEncoder and Attention-based RNNDecoder. The use of embeddings and gru are pretty standard. Other notable model parts involve the use of `F.softmax` after `nn.Linear`, `F.relu` as major activation function, and `dropout = 0.1` to avoid overfitting. More details can be read from the Jupyter notebook in the repository.

One technique that I use is teacher forcing. When teaching forcing is applied, the model is given access to the ground-truth input sequence at each time step, and is expected to use this information to generate the next word in the sequence. This can help the model to stay on track and generate outputs that more closely match the ground-truth input. The teaching forcing ratio is a hyperparameter that determines the extent to which the model should rely on the ground-truth input sequence when generating the next word in the sequence. I use a ratio of 0.5, which I believe seems to provide decreased loss after training on 60 epochs.

I choose a batch size of 256 and train the model at learning rate = 0.02 for 120 epochs, 0.01 for 840 epochs, and 0.005 for 360 epochs. Totally, I train the model for 1320 epochs (more than 15 hours on Colab with 3 alternating accounts), where the model stops to progress and perform better after training.

I do not add any momentum and weight decay as my experiment shows that it only makes the model worse for the first 60 epochs. This might need more further investigation as I believe the momentum should only stabilize the model when the learning rate is lower.

## Evaluation

TODO

## Results

The final model still performs pretty bad

## Conclusion
