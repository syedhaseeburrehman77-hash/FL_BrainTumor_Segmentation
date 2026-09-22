import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss

from utils.metrics import fets_metrics


def train(model, loader, epochs: int, learning_rate: float, device: torch.device) -> float:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    losses = []

    for _ in range(epochs):
        for batch in loader:
            image, label = batch["image"].to(device), batch["label"].long().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(image), label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    totals = {"eval_loss": 0.0, "dice_et": 0.0, "dice_tc": 0.0, "dice_wt": 0.0,
              "hd95_et": 0.0, "hd95_tc": 0.0, "hd95_wt": 0.0}

    for batch in loader:
        image, label = batch["image"].to(device), batch["label"].long().to(device)
        logits = sliding_window_inference(image, (96, 96, 96), 1, model)
        totals["eval_loss"] += loss_fn(logits, label).item()
        for key, value in fets_metrics(logits, label).items():
            totals[key] += value

    return {key: value / max(len(loader), 1) for key, value in totals.items()}