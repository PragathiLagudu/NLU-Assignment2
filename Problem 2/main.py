from utils.dataset import NameDataset
from utils.train import train
from utils.evaluate import novelty_rate, diversity
from models.rnn import VanillaRNN
from models.blstm import BLSTM
from models.attention_rnn import AttentionRNN

# Load dataset
dataset = NameDataset("TrainingNames.txt")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_model(model_class, model_name):
    print(f"\n{'='*20} {model_name} {'='*20}")

    model = model_class(dataset.vocab_size)

    print(f"Trainable Parameters: {count_parameters(model)}")

    # Train
    train(model, dataset)

    #  Different sample size for BLSTM
    target_samples = 50 if model_name == "BLSTM" else 200

    generated = []
    attempts = 0
    max_attempts = 500

    while len(generated) < target_samples and attempts < max_attempts:

        if model_name == "BLSTM":
            name = model.generate(dataset)  
        else:
            name = model.generate(dataset)

        attempts += 1

        #  Relax condition slightly for BLSTM
        if model_name == "BLSTM":
            if len(name) >= 2:
                generated.append(name)
        else:
            if len(name) >= 3:
                generated.append(name)

    print(f"Generated {len(generated)} names in {attempts} attempts")

    #  Handle empty case safely
    if len(generated) == 0:
        print(" No valid names generated")
        generated = ["dummy"]

    print("Sample:", generated[:10])
    print("Novelty:", round(novelty_rate(generated, set(dataset.names)), 3))
    print("Diversity:", round(diversity(generated), 3))


run_model(VanillaRNN, "Vanilla RNN")
run_model(BLSTM, "BLSTM")
run_model(AttentionRNN, "Attention RNN")
