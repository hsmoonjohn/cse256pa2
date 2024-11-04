import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import torch.nn as nn
import torch.optim as optim
import argparse

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerEncoder, TransformerDecoder
from utilities import Utilities
seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to create target mask for the decoder (prevents peeking at future tokens)
def create_target_mask(seq):
    seq_len = seq.size(1)
    mask = torch.tril(torch.ones((seq_len, seq_len), device=seq.device)).unsqueeze(0).unsqueeze(0)
    return mask  # Shape: (1, 1, seq_len, seq_len)

class TransformerClassificationModel(nn.Module):
    def __init__(self, encoder, classifier):
        super(TransformerClassificationModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x, mask):
        encoded_output = self.encoder(x, mask)
        return self.classifier(encoded_output)

class Classifier(nn.Module):
    def __init__(self, embed_size, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(embed_size, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        mask = create_mask(X)
        optimizer.zero_grad()
        outputs = model(X, mask)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def create_mask(x):
    """Creates a mask to ignore padding tokens in the input sequence."""
    return (x != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_length)

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader, criterion):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            mask = create_mask(X)
            outputs = classifier(X, mask)
            loss = criterion(outputs, Y)  # Compute loss for this batch
            total_loss += loss.item()  # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        avg_loss = total_loss / len(data_loader)
        classifier.train()
        return accuracy, avg_loss


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main(part):

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    if part == "part1":

        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=False)

        encoder = TransformerEncoder(embed_size=n_embd, num_layers=n_layer, num_heads=n_head, ff_hidden_dim=n_hidden, dropout=0.1, max_length=block_size, vocab_size=tokenizer.vocab_size).to(device)
        classifier = Classifier(n_embd, n_output).to(device)
        model = TransformerClassificationModel(encoder, classifier).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        print("Starting classification training...")
        for epoch in range(epochs_CLS):
            train_loss = train(model, train_CLS_loader, optimizer, criterion)
            test_accuracy, test_loss = compute_classifier_accuracy(model, test_CLS_loader, criterion)
            print(f"Epoch {epoch+1}/{epochs_CLS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        # Initialize Utilities and run sanity check
        utilities = Utilities(tokenizer, model)
        sample_sentence = "This is a sample sentence to visualize attention maps, which is intentionally made around 32 words to see if the attention map works properly."
        utilities.sanity_check(sample_sentence, block_size=32)
        num_params = count_parameters(encoder)
        print(f"Number of trainable parameters in the TransformerEncoder: {num_params}")


    elif part == "part2":

  
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        vocab_size = tokenizer.vocab_size
        decoder = TransformerDecoder(vocab_size=vocab_size, embed_size=n_embd, num_layers=n_layer, num_heads=n_head, 
                                     ff_hidden_dim=n_hidden, dropout=0.1, max_length=block_size).to(device)
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        print("Starting decoder pretraining on language modeling task...")
        train_losses = []
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            # Move data to device
            xb, yb = xb.to(device), yb.to(device)

            # Generate target mask for the current batch
            trg_mask = create_target_mask(xb)

            # Forward pass through the model
            outputs = decoder(xb, trg_mask=trg_mask)  # Shape: (batch_size, seq_length, vocab_size)

            # Reshape outputs and targets for loss calculation
            loss = criterion(outputs.view(-1, vocab_size), yb.view(-1))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Print training progress and evaluate perplexity at intervals
            if (i + 1) % eval_interval == 0 or i == max_iters - 1:
                avg_loss = sum(train_losses[-eval_interval:]) / len(train_losses[-eval_interval:])
                perplexity = torch.exp(torch.tensor(avg_loss))
                print(f"Iteration {i + 1}/{max_iters} - Training Loss: {avg_loss:.4f} - Perplexity: {perplexity:.2f}")
        utilities = Utilities(tokenizer, decoder)
        sample_sentence = "This is a sample sentence to visualize attention maps, which is intentionally made around 32 words to see if the attention map works properly."
        utilities.sanity_check(sample_sentence, block_size=32, part1=False)

        print("Training complete.")

    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specified part of the assignment")
    parser.add_argument('--part', type=str, choices=['part1', 'part2', 'part3'], required=True, help="Specify which part to run: 'part1' or 'part2' or 'part3'")
    args = parser.parse_args()
    
    main(args.part)
