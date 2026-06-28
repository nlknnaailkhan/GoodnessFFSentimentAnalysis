import os
import time
import pandas as pd
import numpy as np
import torch
import kagglehub
import gensim.downloader as api
from omegaconf import OmegaConf
from collections import defaultdict

# Local project imports
from src.utils import parse_args, preprocess_inputs, update_learning_rate, log_results, print_results, goodness_functions
from src.ffdata import get_data
from src.ffmodel import get_model_and_optimizer

# 1. Config Block
config_dict = {
    "seed": 100,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "input": {
        "path": "datasets",
        "batch_size": 100,
        "num_classes": 10,
        "num_features": 310
    },
    "model": {
        "peer_normalization": 0.03,
        "momentum": 0.9,
        "hidden_dim": 200,
        "num_layers": 3,
        "goodness_type": "log_sum_exp"
    },
    "training": {
        "epochs": 50,
        "learning_rate": 3e-4,
        "weight_decay": 3e-4,
        "momentum": 0.9,
        "downstream_learning_rate": 1e-2,
        "downstream_weight_decay": 3e-3,
        "val_idx": -1,
        "final_test": True
    }
}
opt = OmegaConf.create(config_dict)

def download_and_preprocess_data():
    output_dir = opt.input.path
    if os.path.exists(os.path.join(output_dir, "train_images.npy")):
        print("Dataset files already processed. Skipping download phase.")
        return

    print("Loading Word2Vec embeddings (this may take a minute)...")
    word2vec = api.load("word2vec-google-news-300")
    print("Embeddings loaded successfully!")

    print("Downloading Rotten Tomatoes dataset...")
    path = kagglehub.dataset_download("mrbaloglu/rotten-tomatoes-reviews-dataset")
    print("Path to dataset files:", path)

    os.makedirs(output_dir, exist_ok=True)
    csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
    csv_path = os.path.join(path, csv_file)
    df = pd.read_csv(csv_path)

    raw_labels = df.iloc[:, 0].values
    raw_reviews = df.iloc[:, 1].values

    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(raw_reviews))
    shuffled_reviews = raw_reviews[shuffle_indices]
    shuffled_labels = raw_labels[shuffle_indices]

    print("Vectorizing reviews...")
    embedded_features = []
    for sentence in shuffled_reviews:
        words = str(sentence).lower().split()
        vectors = [word2vec[w] for w in words if w in word2vec]
        mean_vector = np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(300)
        embedded_features.append(mean_vector)

    embedded_features = np.array(embedded_features, dtype=np.float32)
    total_samples = len(embedded_features)
    train_end = int(total_samples * 0.8)
    val_end = int(total_samples * 0.9)

    np.save(os.path.join(output_dir, "train_images.npy"), embedded_features[:train_end])
    np.save(os.path.join(output_dir, "train_labels.npy"), shuffled_labels[:train_end].astype(np.int64))
    np.save(os.path.join(output_dir, "val_images.npy"), embedded_features[train_end:val_end])
    np.save(os.path.join(output_dir, "val_labels.npy"), shuffled_labels[train_end:val_end].astype(np.int64))
    np.save(os.path.join(output_dir, "test_images.npy"), embedded_features[val_end:])
    np.save(os.path.join(output_dir, "test_labels.npy"), shuffled_labels[val_end:].astype(np.int64))
    print("\nSuccess! Transformed total text dataset into clean files.")

def train(opt, model, optimizer):
    start_time = time.time()
    train_loader = get_data(opt, "train")
    num_steps_per_epoch = len(train_loader)

    for epoch in range(opt.training.epochs):
        model.train()
        train_results = defaultdict(float)
        optimizer = update_learning_rate(optimizer, opt, epoch)

        for inputs, labels in train_loader:
            inputs, labels = preprocess_inputs(opt, inputs, labels)
            optimizer.zero_grad()
            scalar_outputs = model(inputs, labels)
            scalar_outputs["Loss"].backward()
            optimizer.step()
            train_results = log_results(train_results, scalar_outputs, num_steps_per_epoch)

        print_results("train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        if epoch == 50:
            validate_or_test(opt, model, "val", epoch=epoch)
    return model

def validate_or_test(opt, model, partition, epoch=None):
    test_time = time.time()
    test_results = defaultdict(float)
    data_loader = get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    print(partition)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = preprocess_inputs(opt, inputs, labels)
            scalar_outputs = model.forward_downstream_classification_model(inputs, labels)
            test_results = log_results(test_results, scalar_outputs, num_steps_per_epoch)

    print_results(partition, time.time() - test_time, test_results, epoch=epoch)
    model.train()

if __name__ == "__main__":
    print("Parsing Configurations...")
    parsed_opt = parse_args(opt)

    print("\nSetting up Data...")
    download_and_preprocess_data()

    print("\nInitializing Model and Optimizer...")
    model, optimizer = get_model_and_optimizer(parsed_opt)

    # Initialize dynamic thresholds
    init_loader = get_data(parsed_opt, "train")
    first_inputs, first_labels = next(iter(init_loader))
    first_inputs, first_labels = preprocess_inputs(parsed_opt, first_inputs, first_labels)

    z_init = torch.cat([first_inputs["pos_images"], first_inputs["neg_images"]], dim=0)
    z_init = z_init.reshape(z_init.shape[0], -1)

    with torch.no_grad():
        z_init = model._layer_norm(z_init)
        z_init = model.model[0](z_init)
        z_init = model.act_fn.apply(z_init)

        goodness_type = getattr(opt.model, 'goodness_type', 'sum_of_squares')
        goodness_fn = goodness_functions.get(goodness_type, goodness_functions['sum_of_squares'])
        raw_goodness = goodness_fn(z_init)
        threshold = torch.mean(raw_goodness).item()

    model.threshold = threshold

    print("\nStarting Training Loop...")
    model = train(parsed_opt, model, optimizer)

    print("\nRunning Final Validation...")
    validate_or_test(parsed_opt, model, "val")

    if parsed_opt.training.final_test:
        print("\nRunning Final Test...")
        validate_or_test(parsed_opt, model, "test")