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

    CONFIG = "./experiments/004.yaml"
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
    best_loss, best_path, last_save_path = 1e10, None, None

    # Weights and biases
    wandb.init(project="APAYN", config=cfg, notes=cfg.get("description", None))
    wandb.save("*.png")  # Write MP4 files immediately to WandB
    wandb.watch(model)

    # Run
    for epoch in range(starting_epoch, n_epochs + 1):
        print(f"\nEpoch {epoch} of {n_epochs}")

        # Cycle
        train_ce_loss, train_iou_loss, train_iou_score = cycle('train', model, dl_train, epoch, optimizer, cfg, scheduler)
        test_ce_loss, test_iou_loss, test_iou_score = cycle('test', model, dl_test, epoch, optimizer, cfg, scheduler)

        # Save state if required
        if tracked_metric == 'crossentropy':
            test_metric = test_ce_loss
        elif tracked_metric == 'jaccard':
            test_metric = test_iou_loss
        elif tracked_metric == 'iou':
            test_metric = test_iou_score
            lowest_best = False

        model_weights = model.module.state_dict() if cfg['training']['data_parallel'] else model.state_dict()
        state = {'epoch': epoch + 1,
                 'model': model_weights,
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler}
        save_name = f"{epoch}_{test_metric:.05f}.pt"
        best_loss, last_save_path = save_state(state, save_name, test_metric, best_loss, cfg, last_save_path, lowest_best=lowest_best)

        # vis
        vis(model, ds_test, cfg, epoch)

    # Save the final epoch
    save_name = f"FINAL_{epoch}_{test_metric:.05f}.pt"
    best_loss, last_save_path = save_state(state, save_name, test_metric, best_loss, cfg, last_save_path, force=True)

