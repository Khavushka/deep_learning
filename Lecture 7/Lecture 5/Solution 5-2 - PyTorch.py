import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader,TensorDataset
from torchtext import datasets, transforms
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm



def vectorize_sequences(sentences,seq_length):
    '''
    Create array of zeros and fill it with fix length reviews. Every review is padded with 0, so they have the same size.
    :param sentences: vectorized IDs based on the vocabulary. e.g. [
    :param seq_length: maximum size of every review.
    :return: np.array of IDs with zeros as padding and seq_length.
    '''
    vectorized_sequence = np.zeros((len(sentences), seq_length),dtype=int)
    for i, review in enumerate(sentences):
        vectorized_sequence[i, :len(review)] = np.array(review)[:seq_length]
    return vectorized_sequence


def tokenize(text):
    '''
    Sets the text in lower case and then returns a list of words.
    :param text: Text to split using spaces as split character.
    :return: Split text into words.
    '''
    return text.lower().split()


def prepare_datasets(train_dataset,validation_dataset):
    '''
    Divides the train and validation dataset into tokenized reviews. Then it counts the tokens (words) from every review to
    generate a vocabulary with the most common words. After that, change the words of the reviews for the IDs in the vocab.
    :param train_dataset:
    :param validation_dataset:
    :return: the padded ID train sequence, formatted train labels, the padded ID validation sequence, formatted validation labels
            and the vocabulary.
    '''

    # Creates a list of tokenized reviews for each dataset.
    tokenized_train_samples = [tokenize(review) for _, review in train_dataset]
    tokenized_validation_samples = [tokenize(review) for _, review in validation_dataset]

    # Defines a Counter that sets a tuple of (word,count) for every word appearance. This iterates first the train samples
        # to get all tokenized reviews, and each tokenized word of the tokenized reviews is added.
    counter = Counter()
    counter.update(token for tokenized_sample in tokenized_train_samples for token in tokenized_sample)

    # Set the labels to [0,1] of each dataset and store it in a separated list.
    encoded_train_labels = [label - 1 for label, _ in train_dataset]
    encoded_validation_labels = [label - 1 for label, _ in validation_dataset]

    # get the first 1000 words to generate a vocabulary.  Then assign id numeration.
    vocab = {word: idx for idx, (word, count) in enumerate(counter.most_common(1000))}

    # Convert tokenized text to sequences of numerical IDs
    train_sequences = [[vocab.get(token, 0) for token in sample] for sample in tokenized_train_samples]
    validation_sequences = [[vocab.get(token, 0) for token in sample] for sample in tokenized_validation_samples]

    # Set the same length for every review so the input size is fixed for the NN.

    padded_train_sequences = vectorize_sequences(train_sequences,seq_length=100)
    padded_validation_sequences = vectorize_sequences(validation_sequences,seq_length=100)

    # Turn them to np.arrays, so it can be processed faster and loaded as a TensorDataset
    return np.array(padded_train_sequences),np.array(encoded_train_labels),\
        np.array(padded_validation_sequences),np.array(encoded_validation_labels),vocab

# Load IMDB dataset and split it to train and validation.
train_dataset = datasets.IMDB(root='./data',split ='train')
validation_dataset = datasets.IMDB(root='./data', split='test')

# Use the prepare_datasets to get the padded ID sequences.
vectorized_train,train_encoded_labels,vectorized_validation,validation_encoded_labels,vocab = prepare_datasets(train_dataset=train_dataset, validation_dataset=validation_dataset)

# Transform the data into a TensorDataset.
train_data = TensorDataset(torch.from_numpy(vectorized_train), torch.from_numpy(train_encoded_labels))
valid_data = TensorDataset(torch.from_numpy(vectorized_validation), torch.from_numpy(validation_encoded_labels))

# Load the processed datasets into shuffled data loaders
batch_size = 64
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
validation_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)


class RNN(nn.Module):

    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, output_dim,drop_prob=0.5):
        super(RNN, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size

        # embedding and RNN layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(input_size=64, hidden_size=self.hidden_dim, num_layers=self.no_layers, batch_first=True,
                          nonlinearity='relu')

        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)

        # Initialize the initial hidden state and adjust the shape of h0
        h0 = torch.zeros(self.no_layers, batch_size, self.hidden_dim)
        out, hn = self.rnn(input=embeds, hx=h0)     # Feed the embedding and h0 to the RNN layer.
        # out contains the hidden state at each time-step, hn contains only the final state
        out = self.fc(out[:, -1])

        # sigmoid function
        sig_out = self.sig(out)
        return sig_out


no_layers = 2
vocab_size = len(vocab) + 1  # Extra 1 for padding
embedding_dim = 64
output_dim = 1
hidden_dim = 256

model = RNN(no_layers, vocab_size, hidden_dim, embedding_dim, output_dim, drop_prob=0.5)
print(model)

lr=0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# function to predict accuracy
def acc(pred, label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()


clip = 5
epochs = 20
valid_loss_min = np.Inf
epoch_tr_loss, epoch_vl_loss = [], []
epoch_tr_acc, epoch_vl_acc = [], []

for epoch in range(epochs):
    train_losses = []
    train_acc = 0.0
    model.train()
    # initialize hidden state
    for inputs, labels in tqdm(train_loader):

        model.zero_grad()
        output = model(inputs)

        # calculate the loss
        loss = criterion(output.squeeze(), labels.float())
        # Perform backpropagation
        loss.backward()
        train_losses.append(loss.item())
        # calculating accuracy
        accuracy = acc(output, labels)
        train_acc += accuracy
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    val_losses = []
    val_acc = 0.0
    model.eval()

    # Do not update the weights, only evaluate the performance at the end of each epoch.
    with torch.no_grad():
        for inputs, labels in tqdm(validation_loader):
            output = model(inputs)
            val_loss = criterion(output.squeeze(), labels.float())
            val_losses.append(val_loss.item())
            accuracy = acc(output, labels)
            val_acc += accuracy

    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = train_acc / len(train_loader.dataset)
    epoch_val_acc = val_acc / len(validation_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)

    print(f'Epoch {epoch + 1}')
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc * 100} val_accuracy : {epoch_val_acc * 100}')
    if epoch_val_loss <= valid_loss_min:
        torch.save(model.state_dict(), 'state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, epoch_val_loss))
        valid_loss_min = epoch_val_loss
    print(25 * '==')

