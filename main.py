import argparse
import os
import yaml
import wandb
import torch
from tqdm import tqdm
from models import DepMamba
from datasets import (
    get_dvlog_dataloader,
    get_lmvd_dataloader,
    get_daicwoz_dataloader,
    get_ours_dataloader,
)
from logzero import logger

# generate config.yaml from config_template.yaml
CONFIG_PATH = "./config/config.yaml"


def parse_args():
    # arguments whose default values are in config.yaml
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description="Train and test a model.")

    # model arguments
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--train_gender", type=str)
    parser.add_argument("--test_gender", type=str)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
    )
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-rn", "--run_num", type=int)
    parser.add_argument("-bs", "--batch_size", type=int)
    parser.add_argument("-lr", "--learning_rate", type=float)
    parser.add_argument("-ds", "--dataset", type=str)
    parser.add_argument("-g", "--gpu", type=str)
    parser.add_argument("-tqdm", "--tqdm_able", type=bool)
    parser.add_argument("-tr", "--train", type=bool)
    parser.add_argument("-d", "--device", type=str, nargs="*")

    # wandb arguments
    parser.add_argument("-wdb", "--if_wandb", type=bool)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--your_name", type=str)

    parser.set_defaults(**config)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args


def train_epoch(
    net,
    train_loader,
    loss_fn,
    optimizer,
    device,
    current_epoch,
    total_epochs,
    tqdm_able,
):
    """One training epoch."""
    net.train()
    sample_count = 0
    running_loss = 0.0
    correct_count = 0

    with tqdm(
        train_loader,
        desc=f"Training epoch {current_epoch}/{total_epochs}",
        leave=False,
        unit="batch",
        disable=tqdm_able,
    ) as pbar:
        for x, y, mask in pbar:
            # logger.info(x.shape,y.shape)
            x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
            y_pred = net(x, mask)

            loss = loss_fn(y_pred, y.to(torch.float32))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sample_count += x.shape[0]
            running_loss += loss.item() * x.shape[0]
            # binary classification with only one output neuron
            pred = (y_pred > 0.0).int()
            correct_count += (pred == y).sum().item()

            pbar.set_postfix(
                {
                    "loss": running_loss / sample_count,
                    "acc": correct_count / sample_count,
                }
            )

    return {
        "loss": running_loss / sample_count,
        "acc": correct_count / sample_count,
    }


def val(net, val_loader, loss_fn, device, tqdm_able):
    """Test the model on the validation / test set."""
    net.eval()
    sample_count = 0
    running_loss = 0.0
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc="Validating", leave=False, unit="batch", disable=tqdm_able
        ) as pbar:
            for x, y, mask in pbar:
                # logger.info(x.shape,y.shape)
                x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
                y_pred = net(x, mask)

                loss = loss_fn(y_pred, y.to(torch.float32))

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]
                # binary classification with only one output neuron
                pred = (y_pred > 0.0).int()
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

                l = running_loss / sample_count
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                accuracy = (TP + TN) / sample_count if sample_count > 0 else 0.0

                pbar.set_postfix(
                    {
                        "loss": l,
                        "acc": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1_score,
                    }
                )

    l = running_loss / sample_count
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (TP + TN) / sample_count if sample_count > 0 else 0.0
    return {
        "loss": l,
        "acc": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
    }


def main():
    args = parse_args()
    logger.info(f"args: {args}")
    args.data_dir = os.path.join(args.data_dir, args.dataset)
    # NOTE: Is run_num needed?
    for i_run in range(args.run_num):
        if args.if_wandb:
            wandb_run_name = f"{args.model}-train_gender_{args.train_gender}-test_gender_{args.test_gender}-{args.your_name}"
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=args,
                name=wandb_run_name,
            )
            args = wandb.config
        # Build Save Dir
        os.makedirs(
            f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_run + 1)}",
            exist_ok=True,
        )
        os.makedirs(
            f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_run + 1)}/samples",
            exist_ok=True,
        )
        os.makedirs(
            f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_run + 1)}/checkpoints",
            exist_ok=True,
        )

        # construct the model
        if args.model == "DepMamba":
            if args.dataset == "lmvd":
                net = DepMamba(**args.mmmamba_lmvd)
            elif args.dataset == "dvlog":
                net = DepMamba(**args.mmmamba_dvlog)
            # TODO: add DAIC-WOZ and our dataset settings for loading DepMamba
            elif args.dataset == "daicwoz":
                # net = DepMamba(**args.mmmamba_daicwoz)
                raise NotImplementedError("DAIC-WOZ is not supported yet")
            elif args.dataset == "ours":
                # net = DepMamba(**args.mmmamba_ours)
                raise NotImplementedError("Our dataset is not supported yet")
            else:
                raise ValueError(f"Unknown dataset: {args.dataset}")
        else:  # if args.model == "MAMBA":
            raise NotImplementedError(
                f"The {args.model} method has not been implemented by this repo"
            )
        net = net.to(args.device[0])
        if len(args.device) > 1:
            net = torch.nn.DataParallel(net, device_ids=args.device)

        # prepare the data
        if args.dataset == "dvlog":
            train_loader = get_dvlog_dataloader(
                args.data_dir, "train", args.batch_size, args.train_gender
            )
            val_loader = get_dvlog_dataloader(
                args.data_dir, "valid", args.batch_size, args.test_gender
            )
            test_loader = get_dvlog_dataloader(
                args.data_dir, "test", args.batch_size, args.test_gender
            )
        elif args.dataset == "lmvd":
            train_loader = get_lmvd_dataloader(
                args.data_dir, "train", args.batch_size, args.train_gender
            )
            val_loader = get_lmvd_dataloader(
                args.data_dir, "valid", args.batch_size, args.test_gender
            )
            test_loader = get_lmvd_dataloader(
                args.data_dir, "test", args.batch_size, args.test_gender
            )
        # TODO: add dataloader for DAIC-WOZ and our dataset
        elif args.dataset == "daicwoz":
            # train_loader = get_daicwoz_dataloader(
            #     args.data_dir, "train", args.batch_size, args.train_gender
            # )
            # val_loader = get_daicwoz_dataloader(
            #     args.data_dir, "valid", args.batch_size, args.test_gender
            # )
            # test_loader = get_daicwoz_dataloader(
            #     args.data_dir, "test", args.batch_size, args.test_gender
            # )
            raise NotImplementedError("DAIC-WOZ is not supported yet")
        elif args.datasets == "ours":
            # train_loader = get_ours_dataloader(
            #     args.data_dir, "train", args.batch_size, args.train_gender
            # )
            # val_loader = get_ours_dataloader(
            #     args.data_dir, "valid", args.batch_size, args.test_gender
            # )
            # test_loader = get_ours_dataloader(
            #     args.data_dir, "test", args.batch_size, args.test_gender
            # )
            raise NotImplementedError("Our dataset is not supported yet")
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        # set other training components
        loss_fn = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

        best_val_acc = -1.0
        # NOTE: why not use best_val_acc?
        best_test_acc = -1.0
        if args.train:
            for epoch in range(args.epochs):
                train_results = train_epoch(
                    net=net,
                    train_loader=train_loader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    device=args.device[0],
                    current_epoch=epoch,
                    total_epochs=args.epochs,
                    tqdm_able=args.tqdm_able,
                )
                val_results = val(
                    net=net,
                    val_loader=val_loader,
                    loss_fn=loss_fn,
                    device=args.device[0],
                    tqdm_able=args.tqdm_able,
                )

                val_acc = (
                    val_results["acc"]
                    + val_results["precision"]
                    + val_results["recall"]
                    + val_results["f1"]
                ) / 4.0
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(
                        net.state_dict(),
                        f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_run + 1)}/checkpoints/best_model.pt",
                    )

                if args.if_wandb:
                    wandb.log(
                        {
                            "loss/train": train_results["loss"],
                            "acc/train": train_results["acc"],
                            "loss/val": val_results["loss"],
                            "acc/val": val_results["acc"],
                            "precision/val": val_results["precision"],
                            "recall/val": val_results["recall"],
                            "f1/val": val_results["f1"],
                        }
                    )

        # upload the best model to wandb website
        # load the best model for testing
        with torch.no_grad():
            net.load_state_dict(
                torch.load(
                    f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_run + 1)}/checkpoints/best_model.pt",
                    map_location=args.device[0],
                )
            )
            net.eval()
            test_results = val(
                net, test_loader, loss_fn, args.device[0], args.tqdm_able
            )
            logger.info("Test results:")
            logger.info(test_results)

            with open(
                f"{args.save_dir}/{args.dataset}_{args.model}_{str(i_run + 1)}.txt", "w"
            ) as f:
                test_result_str = f'Accuracy:{test_results["acc"]}, Precision:{test_results["precision"]}, Recall:{test_results["recall"]}, F1:{test_results["f1"]}, Avg:{(test_results["acc"] + test_results["precision"]+ test_results["recall"]+ test_results["f1"])/4.0}'
                f.write(test_result_str)

    if args.if_wandb:
        # TODO: Consider which models to save.
        # artifact = wandb.Artifact("best_model", type="model")
        # artifact.add_file(
        # f"{args.save_dir}/{args.dataset}_{args.model}_{str(args.run_num)}/checkpoints/best_model.pt"
        # )
        # wandb.log_artifact(artifact)
        wandb.run.summary["acc/best_val_acc"] = best_val_acc
        wandb.run.summary["acc/test_acc"] = test_results["acc"]
        wandb.run.summary["loss/test_loss"] = test_results["loss"]
        wandb.run.summary["precision/test_precision"] = test_results["precision"]
        wandb.run.summary["recall/test_recall"] = test_results["recall"]
        wandb.run.summary["f1/test_f1"] = test_results["f1"]

        wandb.finish()


if __name__ == "__main__":
    main()
