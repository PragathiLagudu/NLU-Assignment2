def novelty_rate(generated, training_set):
    if len(generated) == 0:
        return 0.0  

    new = [n for n in generated if n not in training_set]
    return len(new) / len(generated)


def diversity(generated):
    if len(generated) == 0:
        return 0.0  
    return len(set(generated)) / len(generated)
