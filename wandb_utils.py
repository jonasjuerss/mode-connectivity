import wandb


def init_wandb(args):
    if args.use_wandb:
        wandb.init(project="mode-connectivity", entity="camb-mphil", config=args, tags=["Jonas"])
        return wandb.config
    if args.wandb_log:
        wandb.init(project="mode-connecting-curves", entity="miran-oezdogan")
    return args


def log(*args, **kwargs):
    if wandb.run is not None:
        wandb.log(*args, **kwargs)
