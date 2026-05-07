# Python script to demonstrate Federated learning for keyboard Next-Word prediction
# NB: We're using Pytorch as (TFF, syft, and flwr) had not been updated for Python >= 3.12
# The script simulates federated learning across multiple users (Mueni, Ciku, Kamau, & Bob) using
# synthetic data. It includes training metrics and visualization

# --------------------------------------------------------------------------------------------
# Step 0. Import the required modules
# --------------------------------------------------------------------------------------------
import copy
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict


# --------------------------------------------------------------------------------------------
# Step 1. Generate a synthetic dataset
# --------------------------------------------------------------------------------------------
def generate_sentences(base_phrases, variations=80):
    sentences = []
    fillers = ["please", "today", "now", "quickly", "kindly"]
    endings = ["", "please", "now", "thanks", "ok", "lol"]

    for _ in range(variations):
        phrase = random.choice(base_phrases)
        words = phrase.split()

        if random.random() > 0.5:
            pos = random.randint(0, len(words) - 1)
            words.insert(pos, random.choice(endings))

        words.append(random.choice(endings))
        sentences.append(" ".join(words).strip())

    return sentences


# Base phrases for each user(device)
mueni_base = [
    "hello how are you",
    "how is your day",
    "are you coming today",
    "let us meet later",
    "please call me"
]

ciku_base = [
    "hi how are things",
    "are you doing well",
    "what are you doing",
    "let us catch up",
    "text me later"
]

kamau_base = [
    "hello are you okay",
    "how have you been",
    "uko aje leo",
    "tutaonana later",
    "niko sawa"
]

bob_base = [
    "yo",
    "what's new",
    "hello friend",
    "how do you do",
    "goodbye"
]

# Generate the dataset
data: Dict[str, list[str]] = {
    "Mueni": generate_sentences(mueni_base),
    "Ciku": generate_sentences(ciku_base),
    "Kamau": generate_sentences(kamau_base),
    "Bob": generate_sentences(bob_base),
}


# --------------------------------------------------------------------------------------------
# Step 2. Vocabulary
# --------------------------------------------------------------------------------------------
def build_vocab(sentences: List[str]) -> Dict[str, int]:
    words = []
    for s in sentences:
        words.extend(s.split())

    vocab = {w: n + 1 for n, w in enumerate(set(words))}  # Use set to filter duplicates
    vocab["<PAD>"] = 0
    return vocab


all_sentences = sum(data.values(), [])
vocab = build_vocab(all_sentences)
vocab_size = len(vocab)


# --------------------------------------------------------------------------------------------
# Step 3. Dataset
# --------------------------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, sentences, vocab, seq_len=3):
        self.data = []

        for sentence in sentences:
            tokens = [vocab[w] for w in sentence.split()]
            for n in range(len(tokens) - seq_len):
                self.data.append((tokens[n: n + seq_len], tokens[n + seq_len]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)


# --------------------------------------------------------------------------------------------
# Step 4. The model
# --------------------------------------------------------------------------------------------
class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=16, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# --------------------------------------------------------------------------------------------
# Step 5. Training & Evaluation
# --------------------------------------------------------------------------------------------
def train(model, loader, epoch=1):
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epoch):
        for x, y in loader:
            optimiser.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimiser.step()


def compute_loss(model, loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss = 0

    with torch.no_grad():
        for x, y in loader:
            loss += criterion(model(x), y).item()
        return loss / len(loader)


def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            preds = model(x)
            _, predicted = torch.max(preds, dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    return correct / total


# --------------------------------------------------------------------------------------------
# Step 6. Federating averaging
# --------------------------------------------------------------------------------------------
def federating_average(models):
    global_model = copy.deepcopy(models[0])
    global_dict = global_model.state_dict()

    for key in global_dict:
        global_dict[key] = torch.stack(
            [m.state_dict()[key].float() for m in models]
        ).mean(dim=0)

    global_model.load_state_dict(global_dict)
    return global_model


# --------------------------------------------------------------------------------------------
# Step 7. DataLoaders
# --------------------------------------------------------------------------------------------
client_loaders = {
    name: DataLoader(TextDataset(sentences, vocab), batch_size=4, shuffle=True)
    for name, sentences in data.items()
}


# --------------------------------------------------------------------------------------------
# Step 8. Federated learning
# --------------------------------------------------------------------------------------------
def federate_training(rounds=5):
    global_model = LSTM(vocab_size)

    losses, accuracies = [], []

    for r in range(rounds):
        print(f"\n==== Round {r + 1} ====")

        local_models = []
        local_losses = []
        local_accuracies = []

        for name, loader in client_loaders.items():
            print(f"Training on {name}'s device")

            local_model = copy.deepcopy(global_model)
            train(local_model, loader)

            # ✅ compute loss INLINE (no separate function → avoids SymPy bug)
            criterion = nn.CrossEntropyLoss()
            local_model.eval()
            total_loss = 0

            with torch.no_grad():
                for x, y in loader:
                    preds = local_model(x)
                    total_loss += criterion(preds, y).item()

            loss = total_loss / len(loader)

            # accuracy (this function is safe)
            accuracy = evaluate_accuracy(local_model, loader)

            local_losses.append(loss)
            local_accuracies.append(accuracy)
            local_models.append(local_model)

        # ✅ aggregate AFTER all clients trained
        global_model = federating_average(local_models)

        avg_loss = sum(local_losses) / len(local_losses)
        avg_accuracy = sum(local_accuracies) / len(local_accuracies)

        losses.append(avg_loss)
        accuracies.append(avg_accuracy)

        print(f"Loss: {avg_loss:.3f}, Accuracy: {avg_accuracy:.3f}")

    return global_model, losses, accuracies

# --------------------------------------------------------------------------------------------
# Step 9. Prediction
# --------------------------------------------------------------------------------------------
def predict_next_words(model, text, top_k=3):
    model.eval()
    tokens = text.split()[-3:]
    indices = [vocab.get(w, 0) for w in tokens]

    x = torch.tensor(indices).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)

    top_indices = torch.topk(top_k, dim=1).indices.squeeze().tolist()
    inv_vocab = {i: w for w, i in vocab.items()}

    return [inv_vocab[i] for i in top_indices]

# --------------------------------------------------------------------------------------------
# Step 10. Plot metrics
# --------------------------------------------------------------------------------------------
def plot_metrics(losses, accuracies):
    rounds = range(1, len(losses) + 1)

    plt.figure()
    plt.plot(rounds, losses)
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title("Loss vs Round")
    plt.show()

    plt.figure()
    plt.plot(rounds, accuracies)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Round")
    plt.show()

# --------------------------------------------------------------------------------------------
# Step 11. Run the script
# --------------------------------------------------------------------------------------------
if __name__ == "__main__":
    global_model, losses, accuracies = federate_training()
    plot_metrics(losses, accuracies)

    print("\nPredictions")
    print("How are you ->", predict_next_words(global_model,"how are you"))
    print("hello how are ->", predict_next_words(global_model, "hello how are"))