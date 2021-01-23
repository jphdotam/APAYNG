import wandb

from torch.utils.data import DataLoader

from lib.vis import vis
from lib.models import load_model
from lib.config import load_config
from lib.datasets import APAYNSliceDataset
from lib.optimizers import load_optimizer
from lib.transforms import load_transforms
from lib.training import cycle, save_state


if __name__ == "__main__":

    CONFIG = "./experiments/008.yaml"
    cfg = load_config(CONFIG)

    bs_train, bs_test, n_workers = cfg['training']['batch_size_train'], cfg['training']['batch_size_test'], cfg['training']['n_workers']
    n_epochs = cfg['training']['n_epochs']
    tracked_metric, lowest_best = cfg['training']['tracked_metric'], True
    transforms_train, transforms_test = load_transforms(cfg)

    # Data
    ds_train = APAYNSliceDataset(cfg, 'train', transforms_train)
    ds_test = APAYNSliceDataset(cfg, 'test', transforms_test)

    dl_train = DataLoader(ds_train, bs_train, shuffle=True, num_workers=n_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, bs_test, shuffle=False, num_workers=n_workers, pin_memory=True)

    # Model
    model, starting_epoch, state = load_model(cfg)
    optimizer, scheduler = load_optimizer(model, cfg, state, steps_per_epoch=len(dl_train))

    # Training
    best_metric, best_path, last_save_path = 0 if tracked_metric == 'iou' else 1e10, None, None

    # Weights and biases
    wandb.init(project="APAYN", config=cfg, notes=cfg.get("description", None))
    wandb.save("*.png")  # Write MP4 files immediately to WandB
    wandb.watch(model)

    # Run
    for epoch in range(starting_epoch, n_epochs + 1):
        print(f"\nEpoch {epoch} of {n_epochs}")

        # Cycle
        train_ce_loss, train_jaccard_loss, train_iou, train_hybriddice_loss, train_lovasz_loss = cycle('train', model, dl_train, epoch, optimizer, cfg, scheduler)
        test_ce_loss, test_jaccard_loss, test_iou, test_hybriddice_loss, test_lovasz_loss = cycle('test', model, dl_test, epoch, optimizer, cfg, scheduler)

        # Save state if required
        if tracked_metric == 'crossentropy':
            test_metric = test_ce_loss
        elif tracked_metric == 'jaccard':
            test_metric = test_jaccard_loss
        elif tracked_metric == 'dice_ce_hybrid':
            test_metric = test_hybriddice_loss
        elif tracked_metric == 'lovasz':
            test_metric = test_lovasz_loss
        elif tracked_metric == 'iou':
            test_metric = test_iou
            lowest_best = False
        else:
            raise ValueError(f"Unknown tracked metric {tracked_metric}")

        model_weights = model.module.state_dict() if cfg['training']['data_parallel'] else model.state_dict()
        state = {'epoch': epoch + 1,
                 'model': model_weights,
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler}
        save_name = f"{epoch}_{test_metric:.05f}.pt"
        best_metric, last_save_path = save_state(state, save_name, test_metric, best_metric, cfg, last_save_path, lowest_best=lowest_best)

        # vis
        vis(model, ds_test, cfg, epoch)

    # Save the final epoch
    save_name = f"FINAL_{epoch}_{test_metric:.05f}.pt"
    best_metric, last_save_path = save_state(state, save_name, test_metric, best_metric, cfg, last_save_path, force=True)

