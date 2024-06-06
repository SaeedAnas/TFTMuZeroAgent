import pytest
import time
from tqdm import tqdm

from Simulator.batch_generator import BatchGenerator, TokenBatchGenerator
from Models.representation_model import TraitPredictor
import torch

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# --- Train Utils --- #


def load_batch(generator, batch_length):
    return (
        torch.as_tensor(x, dtype=torch.float32).to(device)
        for x in generator.generate_batch(batch_length)
    )


def train(generator, model, loss_fn, optimizer, steps, batch_length):
    model.train()
    print(f"Training for {steps} steps with batch length {batch_length}:")
    pbar = tqdm(range(steps))
    for step in pbar:
        X, y = load_batch(generator, batch_length)

        # Forward
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        if step % 10 == 0:
            pbar.set_description(f"Loss: {loss.item()}")
        pbar.update(1)
    pbar.close()


def evaluate(generator, model, loss_fn, test_steps, batch_length):
    model.eval()
    print(f"Testing for {test_steps} steps with batch length {batch_length}:")
    test_loss, correct = 0, 0
    with torch.no_grad():
        for _ in tqdm(range(test_steps)):
            X, y = load_batch(generator, batch_length)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred_labels = torch.nn.functional.one_hot(
                torch.argmax(pred, dim=-1), num_classes=5
            )
            correct += (pred_labels == y).all(dim=-1).all(dim=-1).sum().item()
    test_loss /= test_steps
    correct /= test_steps * batch_length
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")


@pytest.mark.skip
def test_batch_generator():
    b = TokenBatchGenerator()
    start = time.time()
    traits, labels = b.generate_batch(1)
    print(f"Time taken: {time.time() - start}")
    print(traits.shape)
    print(labels.shape)


@pytest.mark.skip
def test_network():
    b = TokenBatchGenerator()
    model = TraitPredictor().to(device)
    print(model)
    X, y = load_batch(b, 2)
    print(X, y)
    pred = model(X)
    print(pred)


# @pytest.mark.skip
def test_training():
    # --- Config --- #
    epochs = 10
    steps = 100
    test_steps = 20
    batch_length = 2

    # --- Init --- #
    b = TokenBatchGenerator()
    model = TraitPredictor().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- Train --- #
    for _ in range(epochs):
        train(b, model, loss_fn, optimizer,
              steps=steps, batch_length=batch_length)
        evaluate(b, model, loss_fn, test_steps=test_steps,
                 batch_length=batch_length)
