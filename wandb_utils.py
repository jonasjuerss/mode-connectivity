import wandb


def init_wandb(args):
    if args.use_wandb:
        wandb.init(project="mode-connectivity", entity="vlamir", config=args, tags=["Viktor"])
        return wandb.config
    return args


def log(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)
