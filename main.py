import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from preprocessing import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1.) Data

# 1.1.) Path
path = ""  # <-- insert project directory here

# word_embedding_data_path = path + "/data/WordEmbedding_Data_Tiny.txt"
# word_embedding_data_path = path + "/data/WordEmbedding_Data.txt"

# 1.2.) Load Data
word_embedding_data_tiny = []

for line in open(word_embedding_data_path, encoding="utf8"):
    word_embedding_data_tiny.append(line.split())

# 2.) Preprocessing
X = []
pre_corpus = []
corpus = []

pre_corpus = preprocess_text(word_embedding_data_tiny)

for sentence in pre_corpus:
    for e in sentence:
        corpus.append(e)

hot_encoder = OneHotEncoder()
X = np.array(corpus)
hot_encoder.fit(X.reshape(-1, 1))

# 3) Modeling / Embedding
class Embedding(nn.Module):
    def __init__(self, input_shape):
        super(Embedding, self).__init__()
        self.lay1 = nn.Linear(input_shape, 20)
        self.lay2 = nn.Linear(20, input_shape)

    def forward(self, x):
        x = self.lay1(x)
        x = self.lay2(x)
        return F.log_softmax(x, dim=1)

    def get_embedding(self, x):
        return self.lay1(x)


network = Embedding(len(set(corpus)))
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

loss_list = []


def train(epoch, network):
    network.train()
    z = 0
    sentences = len(pre_corpus)
    for sentence in pre_corpus:
        for targetword in sentence:
            targetword = hot_encoder.transform(
                np.array([targetword]).reshape(1, -1)
            ).toarray()
            targetword = torch.tensor(targetword).float()
            for contextword in sentence:
                contextword = hot_encoder.transform(
                    np.array([contextword]).reshape(1, -1)
                ).toarray()
                contextword = torch.tensor(contextword).float()

                if torch.all(
                    torch.eq(targetword.clone().detach(), contextword.clone().detach())
                ):
                    pass

                output = network(targetword)
                loss = F.nll_loss(output, torch.argmax(contextword, dim=1))
                loss.backward()
                loss_list.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()

        z += 1
        print(
            "Train Epoch: {} \t [{:.0f}%] \t sentence: {}/{} ".format(
                epoch, (z / sentences) * 100, z, sentences
            )
        )


n_epochs = 10
for i in range(n_epochs):
    train(i, network)
    print(f"Epoch {i} finished! Average loss: {sum(loss_list) / len(loss_list)} \n")
    loss_list.clear()


word_embeddings = []
for word in set(corpus):
    word = hot_encoder.transform(np.array([word]).reshape(1, -1)).toarray()
    word = torch.tensor(word).float()
    output = network.get_embedding(word)
    output = output.detach().numpy()
    word_embeddings.append(output)


word_embeddings = np.array(word_embeddings)
word_embeddings = np.squeeze(word_embeddings, axis=1)
X_embedded = TSNE(n_components=2).fit_transform(word_embeddings)
ax = sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1])

i = 0
for word in set(corpus):
    ax.text(X_embedded[i, 0], X_embedded[i, 1], word, size=8, zorder=99)
    i += 1

plt.show()
