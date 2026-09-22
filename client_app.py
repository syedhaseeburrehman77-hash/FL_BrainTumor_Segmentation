import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from monai.utils import set_determinism

from datasets_loaders.registry import get_partition_loaders
from models.unet import build_model
from task import evaluate, train

app = ClientApp()

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loaders(context: Context):
    partition_id = int(context.node_config["partition-id"])

    set_determinism(seed=int(context.run_config["seed"]) + partition_id)

    return get_partition_loaders(
        context.run_config,
        partition_id,
    )

@app.train()
def train_client(message: Message, context: Context) -> Message:
    """Train one simulated FeTS institution."""

    device = get_device()
    model = build_model().to(device)
    model.load_state_dict(message.content["arrays"].to_torch_state_dict())

    train_loader, _ = get_loaders(context)

    loss = train(
        model=model,
        loader=train_loader,
        epochs=int(context.run_config["local-epochs"]),
        learning_rate=float(message.content["config"]["lr"]),
        device=device,
    )
    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(model.state_dict()),
                "metrics": MetricRecord(
                    {
                        "train_loss": loss,
                        "num-examples": len(train_loader.dataset),
                    }
                ),
            }
        ),
        reply_to=message,
    )

@app.evaluate()
def evaluate_client(message: Message, context: Context) -> Message:
    """Evaluate the received global model on one institution's validation data."""

    device = get_device()
    model = build_model().to(device)
    model.load_state_dict(message.content["arrays"].to_torch_state_dict())

    _, validation_loader = get_loaders(context)
    metrics = evaluate(model, validation_loader, device)
    metrics["num-examples"] = len(validation_loader.dataset)

    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(metrics),
            }
        ),
        reply_to=message,
    )