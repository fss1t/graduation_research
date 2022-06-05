import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from pathlib import Path
import json
from attrdict import AttrDict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as multiprocessing
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parents[1]))
    sys.path.append(str(Path(__file__).parents[2]))

from train.tool.dataset import SpeF0Dataset
from train.tool.schedule import CosExp
from common.model.jdcnet import JDCNet

torch.backends.cudnn.benchmark = True


def train(path_dir_list=Path("../dataset/list"),
          path_dir_param=Path("../dataset/param"),
          path_dir_checkpoint=Path("./checkpoint"),
          path_dir_log=Path("./log"),
          path_config_data="../../HiFiGAN/config_v1.json",
          path_config_train="./config.json",):
    os.chdir(os.path.dirname(__file__))  # cd .
    print("--- train ---")

    # prepare directory

    path_dir_checkpoint.mkdir(exist_ok=1)
    path_dir_log.mkdir(exist_ok=1)

    # load config

    with open(path_config_data, "r") as js:
        hd = json.loads(js.read())
    hd = AttrDict(hd)

    with open(path_config_train, "r") as js:
        h = json.loads(js.read())
    h = AttrDict(h)

    # prepare device

    if multiprocessing.get_start_method() == 'fork' and h.num_workers != 0:
        multiprocessing.set_start_method('spawn', force=True)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(h.seed)

    # prepare dataset

    dataset_train = SpeF0Dataset(path_dir_list / "wav_train.txt",
                                 path_dir_param / "norm.json",
                                 hd.n_fft,
                                 hd.win_size,
                                 hd.hop_size,
                                 hd.num_mels,
                                 hd.sampling_rate,
                                 hd.fmin,
                                 hd.fmax,
                                 h.f0_floor,
                                 h.f0_ceil,
                                 h.segment_size,
                                 h.seed)
    dataset_valid = SpeF0Dataset(path_dir_list / "wav_valid.txt",
                                 path_dir_param / "norm.json",
                                 hd.n_fft,
                                 hd.win_size,
                                 hd.hop_size,
                                 hd.num_mels,
                                 hd.sampling_rate,
                                 hd.fmin,
                                 hd.fmax,
                                 h.f0_floor,
                                 h.f0_ceil,
                                 h.segment_size_valid,
                                 None)

    dataloader_train = DataLoader(dataset_train,
                                  h.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=h.num_workers,
                                  pin_memory=True)
    dataloader_valid = DataLoader(dataset_valid,
                                  h.batch_size_valid,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=h.num_workers,
                                  pin_memory=True)

    # prepare model

    recognizer = JDCNet()
    recognizer = recognizer.to(h.device)

    path_cp = path_dir_checkpoint / "recognizer_latest.cp"
    if path_cp.exists():
        cp = torch.load(path_cp, map_location=lambda storage, loc: storage)
        recognizer.load_state_dict(cp)
        del cp
        print(f"loaded {path_cp}")

    # prepare optimizer

    optimizer = torch.optim.RAdam(recognizer.parameters(),
                                  h.lr,
                                  h.betas,
                                  h.eps,
                                  h.weight_decay)

    cosexp = CosExp(h.epoch_warmup * len(dataloader_train),
                    h.epoch_switch * len(dataloader_train),
                    h.weight_lr_initial,
                    h.weight_lr_final)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosexp)

    path_cp = path_dir_checkpoint / "state_latest.cp"
    if path_cp.exists():
        cp = torch.load(path_cp, map_location=lambda storage, loc: storage)
        optimizer.load_state_dict(cp["optimizer"])
        scheduler.load_state_dict(cp["scheduler"])
        epoch = cp["epoch"]
        step = cp["step"]
        loss_best = cp["loss_best"]
        del cp
        print(f"loaded {path_cp}")
    else:
        epoch = 0
        step = 0
        loss_best = 1.0

    # prepare loss function

    f_sl1loss = torch.nn.SmoothL1Loss()
    f_bceloss = torch.nn.BCELoss()

    # prepare tensorboard
    sw = SummaryWriter(path_dir_log)

    # start training

    while(epoch < h.epochs):
        epoch += 1
        print(f"--- epoch {epoch} train ---")

        # train

        for batch in tqdm(dataloader_train):
            step += 1

            # forward

            spe, f0, vuv = [item.to(h.device) for item in batch]

            f0_h, vuv_h = recognizer(spe)

            loss_f0 = f_sl1loss(f0_h * vuv, f0)
            loss_vuv = f_bceloss(vuv_h, vuv)
            loss = loss_f0 + h.weight_loss_vuv * loss_vuv

            # backward

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # write log
            sw.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            scheduler.step()

            sw.add_scalar("train/f0_loss", loss_f0, step)
            sw.add_scalar("train/vuv_loss", loss_vuv, step)

        # valid

        if epoch % h.valid_interval == 0:
            print(f"--- epoch {epoch} valid ---")

            recognizer.eval()
            with torch.no_grad():
                loss_f0 = 0.0
                loss_vuv = 0.0
                for batch in tqdm(dataloader_valid):
                    # forward

                    spe, f0, vuv = [item.to(h.device) for item in batch]

                    f0_h, vuv_h = recognizer(spe)

                    loss_f0 = f_sl1loss(f0_h * vuv, f0)
                    loss_vuv = f_bceloss(vuv_h, vuv)
                loss_f0 /= len(dataloader_valid)
                loss_vuv /= len(dataloader_valid)
                loss = loss_f0 + h.weight_loss_vuv * loss_vuv

                # write log
                sw.add_scalar("valid/f0_loss", loss_f0, step)
                sw.add_scalar("valid/vuv_loss", loss_vuv, step)

                # save state

                torch.save(
                    recognizer.state_dict(),
                    path_dir_checkpoint / "recognizer_latest.cp")
                torch.save(
                    {"optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "epoch": epoch,
                     "step": step,
                     "loss_best": loss_best},
                    path_dir_checkpoint / "state_latest.cp")

                print("saved recognizer_latest.cp and state_latest.cp")

                if loss <= loss_best:
                    loss_best = loss
                    torch.save(
                        recognizer.state_dict(),
                        path_dir_checkpoint / "recognizer_best.cp")
                    print("saved recognizer_best.cp")

            recognizer.train()


if __name__ == "__main__":
    train()
