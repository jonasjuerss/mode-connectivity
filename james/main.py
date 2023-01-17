import pickle, ssl, logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytorch_lightning as pl

import models, curves, utils1, utils, preresnet, resnet

DATASET = datasets.CIFAR10
TEST_ITEMS = 50_000
BATCH_SIZE = 128
WORKER_COUNT = 6
EPOCHS = 120

LEARNING_RATE=0.1
LR_GAMMA=0.1
L2_REG = 5e-4
MOMENTUM = 0.9


PARALLEL_MODELS = 1
MODEL = resnet.ResNet18 if DATASET == datasets.CIFAR10 else resnet.ResNet14MNIST

CURVE_BENDS = 3
CURVE_NUM_SAMPLES = 61

STEP_MODES = 2
STEP_BAD_MODES = 2


CODE_CHECK = False
if CODE_CHECK:
    TEST_ITEMS = 5000
    WORKER_COUNT = 0
    EPOCHS = 1
    CURVE_NUM_SAMPLES = 2

LOADER_ARGS = {"batch_size":BATCH_SIZE, "num_workers":WORKER_COUNT, "persistent_workers":WORKER_COUNT>0}
TRAINER_ARGS = {"accelerator":"gpu", "devices":"auto", "max_epochs":EPOCHS, "precision":16}


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        if PARALLEL_MODELS > 1:
            self.model = models.ParallelModel(PARALLEL_MODELS, MODEL, 10, **MODEL.kwargs)
        else:
            self.model = MODEL.base(10, **MODEL.kwargs)
        self.loss = nn.CrossEntropyLoss()
        self.mode = None

    def forward(self, x):
        return self.model(x)

    def process_batch(self, batch):
        x, y = batch
        if PARALLEL_MODELS > 1:
            y = torch.cat([y for _ in range(PARALLEL_MODELS)], dim=0)
        return self(x), y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.process_batch(batch)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss.item(), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.process_batch(batch)
        loss = self.loss(y_hat, y)
        correct = (y_hat.argmax(1) == y).type(torch.float).sum().item()
        acc = 100*correct / y.size(dim=0)
        self.log("val_loss", loss)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        y_hat, y = self.process_batch(batch)
        loss = self.loss(y_hat, y)
        correct = (y_hat.argmax(1) == y).type(torch.float).sum().item()
        acc = 100*correct / y.size(dim=0)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        
    def configure_optimizers(self):
        optimiser = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE*PARALLEL_MODELS, momentum=MOMENTUM, weight_decay=L2_REG)
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.StepLR(optimiser, step_size=max(1,EPOCHS//3), gamma=LR_GAMMA),
            "interval": "epoch"
        }
        return {"optimizer":optimiser, "lr_scheduler":scheduler_dict}



class LitModelConnect(pl.LightningModule):
    def __init__(self, start_model = None, end_model = None, num_bends=CURVE_BENDS):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.model = curves.CurveNet(10, curves.PolyChain, MODEL.curve, num_bends, architecture_kwargs=MODEL.kwargs)
        self.t = None
        self.update_bn = False

        # Initialise curve weights
        if start_model != None:
            self.model.import_base_parameters(start_model.model, 0)
            self.model.import_base_parameters(end_model.model, num_bends - 1)
            self.model.init_linear()

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def regulariser(self):
        return 0.5 * L2_REG * self.model.l2

    def process_batch(self, batch):
        x, y = batch
        return self(x, t=self.t), y

    def set_t(self, t):
        self.t = t

    def training_step(self, batch, batch_idx):
        y_hat, y = self.process_batch(batch)
        loss = self.loss(y_hat, y) + self.regulariser()
        self.log("train_loss", loss.item(), on_epoch=True)
        return loss

    def on_test_start(self):
        if self.update_bn:
            self.model.train()

    def test_step(self, batch, batch_idx):
        y_hat, y = self.process_batch(batch)
        nll = self.loss(y_hat, y)
        loss = nll + self.regulariser()
        correct = (y_hat.argmax(1) == y).type(torch.float).sum().item()
        acc = 100*correct / y.size(dim=0)

        self.log("test_nll", nll)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        
    def configure_optimizers(self):
        optimiser = torch.optim.SGD(
            filter(lambda param: param.requires_grad, self.parameters()),
            lr=LEARNING_RATE,
            momentum=MOMENTUM
        )
        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.StepLR(optimiser, step_size=max(1,EPOCHS//3), gamma=LR_GAMMA),
            "interval": "epoch"
        }
        return {"optimizer":optimiser, "lr_scheduler":scheduler_dict}

def update_bn(model, loader):
    if not utils.check_bn(model): return

    bn_trainer = pl.Trainer(logger=False,**TRAINER_ARGS)
    model.update_bn = True
    bn_trainer.test(model, loader, verbose=False)
    model.update_bn = False

def testCurve(model, trainer, test_loader, train_loader=None):
    ts = np.linspace(0.0, 1.0, CURVE_NUM_SAMPLES)
    if train_loader == None:
        train_loader = test_loader

    # BN has momentum so iter a few times to warm up
    model.set_t(0.0)
    for _ in range(3):
        update_bn(model, train_loader)

    # Test and compute max stats
    max_loss = -1
    for t in ts:
        model.set_t(t)
        update_bn(model, train_loader)

        metrics = trainer.test(model, test_loader)
        max_loss = max(max_loss, metrics[0]["test_loss"])

    return max_loss



if __name__ == "__main__":

    # Select training device
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        LOADER_ARGS["pin_memory"] = True
    device = "cuda" if use_cuda else "cpu"
    print(f"Training on {device}")

    # Setup logging
    logging.getLogger("lightning").setLevel(logging.ERROR)
    def genLogger(log, path):
        return pl.loggers.CSVLogger(path, name=log)

    # Setup checkpointing
    checkpoint = pl.callbacks.ModelCheckpoint(
        #dirpath="checkpoints",
        #save_last=True,
        save_top_k=0,
        save_weights_only=True
    )

    # Setup progress bar
    progress_bar = pl.callbacks.RichProgressBar()
    TRAINER_ARGS["callbacks"] = [checkpoint, progress_bar]


    # ------------------------
    # Prepare Datasets / Loaders
    # ------------------------

    # CIFAR10 has an expired cert
    ssl._create_default_https_context = ssl._create_unverified_context

    # Too many files open with file_descriptor strategy
    torch.multiprocessing.set_sharing_strategy('file_system')

    transform = []
    # if DATASET == datasets.CIFAR10:
    #     transform += [transforms.Grayscale()]
    transform += [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]

    transform = transforms.Compose(transform)

    train_data = DATASET(
        root=".data",
        train=True,
        download=True,
        transform=transform
    )

    # Truncate data for quick checks
    if TEST_ITEMS < 50_000:
        train_data = torch.utils.data.Subset(train_data, range(TEST_ITEMS))

    adverse_data = utils1.NoiseDataset(train_data)

    test_data = DATASET(
        root=".data",
        train=False,
        download=True,
        transform=transform
    )

    # train_data = utils1.GPUDataset(train_data, "cuda")
    # adverse_data = utils1.GPUDataset(adverse_data, "cuda")
    # test_data = utils1.GPUDataset(test_data, "cuda")

    train_loader   = DataLoader(train_data, shuffle=True, **LOADER_ARGS)
    adverse_loader = DataLoader(adverse_data, shuffle=True, **LOADER_ARGS)
    LOADER_ARGS["batch_size"] = 1024
    test_loader    = DataLoader(test_data, **LOADER_ARGS)
    curve_loader   = DataLoader(train_data, **LOADER_ARGS)



    # ------------------------
    # Generate Data
    # ------------------------

    

    # Load state
    state = {"minima":[], "paths":[]}

    try:
        state_file = open('state.p', 'rb')
        state = pickle.load(state_file)
        state_file.close()
    except FileNotFoundError:
        pass

    # ------------------------
    # Generate Minima
    # ------------------------
    
    # Generate good minima
    good_minima = []
    for _ in range(STEP_MODES):
        model = LitModel()
        model.mode = "train"
        path=f"modes/{len(state['minima'])}"
        trainer = pl.Trainer(logger=genLogger("train", path),**TRAINER_ARGS)
        trainer.fit(model, train_loader, test_loader)
        metrics = trainer.test(model, test_loader)[0]
        trainer.save_checkpoint(path+"/model.ckpt")

        record = {"idx":len(state["minima"]), "type":"good", "path":path, "loss":metrics["test_loss"], "acc":metrics["test_acc"]}
        good_minima.append(record)
        state["minima"].append(record)

    # Generate adversarial init
    model = LitModel()
    model.mode = "adverse"
    trainer = pl.Trainer(logger=genLogger("adverse", "modes"), **TRAINER_ARGS)
    trainer.fit(model, adverse_loader)
    trainer.save_checkpoint("modes/adverse.ckpt")

    # Generate bad minima
    bad_minima = []
    for _ in range(STEP_BAD_MODES):
        model = LitModel.load_from_checkpoint("modes/adverse.ckpt")
        model.mode = "train"
        path=f"modes/{len(state['minima'])}"
        trainer = pl.Trainer(logger=genLogger("train", path), **TRAINER_ARGS)
        trainer.fit(model, train_loader, test_loader)
        metrics = trainer.test(model, test_loader)[0]
        trainer.save_checkpoint(path+"/model.ckpt")
        
        record = {"idx":len(state["minima"]), "type":"bad", "path":path, "loss":metrics["test_loss"], "acc":metrics["test_acc"]}
        bad_minima.append(record)
        state["minima"].append(record)

    # ------------------------
    # Generate Curves
    # ------------------------

    new_curves = []
    # Connect good minima to existing
    for mode in good_minima:
        if mode["idx"] == 0: continue

        other = np.random.randint(mode["idx"])
        while state["minima"][other]["type"] != "good":
            other = np.random.randint(mode["idx"])

        new_curves.append((mode["idx"], other))

    # Connect a bad minimum to a good minimum
    other = np.random.randint(bad_minima[0]["idx"])
    while state["minima"][other]["type"] != "good":
        other = np.random.randint(bad_minima[0]["idx"])
    new_curves.append((bad_minima[0]["idx"], other))

    # Curve directly between bad minima
    new_curves.append((bad_minima[0]["idx"], bad_minima[1]["idx"]))

    # Random new curves
    start = np.random.randint(len(state["minima"]))
    other = np.random.randint(len(state["minima"]))
    while start == other:
        other = np.random.randint(len(state["minima"]))
    new_curves.append((start, other))

    start = np.random.randint(len(state["minima"]))
    other = np.random.randint(len(state["minima"]))
    while start == other:
        other = np.random.randint(len(state["minima"]))
    new_curves.append((start, other))
    

    for start_idx, end_idx in new_curves:
        start = LitModel.load_from_checkpoint(state["minima"][start_idx]["path"] + "/model.ckpt")
        end = LitModel.load_from_checkpoint(state["minima"][end_idx]["path"] + "/model.ckpt")
        path=f"curves/{len(state['paths'])}"

        linear = LitModelConnect(start, end, 2)
        trainer = pl.Trainer(logger=genLogger("linear", path), **TRAINER_ARGS)
        lin_loss = testCurve(linear, trainer, curve_loader, test_loader)
        trainer.save_checkpoint(path + "/linear.ckpt")

        curve = LitModelConnect(start, end, CURVE_BENDS)
        trainer = pl.Trainer(logger=genLogger("curve", path), **TRAINER_ARGS)
        trainer.fit(curve, train_loader)
        trainer.save_checkpoint(path + "/curve.ckpt")
        curve_loss = testCurve(curve, trainer, curve_loader, test_loader)
        curve_test_loss = testCurve(curve, trainer, test_loader, test_loader)

        record = {"idx":len(state["paths"]), "path":path, "start":start_idx, "end":end_idx, "lin_loss":lin_loss, "curve_loss":curve_loss, "curve_test_loss":curve_test_loss}
        state["paths"].append(record)

    # Save state
    pickle.dump(state, open('state.p', 'wb'))