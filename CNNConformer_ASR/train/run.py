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

from train.tool.dataset import SpeLabDataset
from train.tool.schedule import CosExp
from train.tool.cer import CER
from common.tool.get_num_class import get_num_class
from common.conformer.model import Conformer


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

    dataset_train = SpeLabDataset(path_dir_list / "wav_train.txt",
                                  path_dir_list / "lab_train.txt",
                                  path_dir_param / "norm.json",
                                  path_dir_param / "phoneme.json",
                                  hd.n_fft,
                                  hd.win_size,
                                  hd.hop_size,
                                  hd.num_mels,
                                  hd.sampling_rate,
                                  hd.fmin,
                                  hd.fmax,
                                  h.segment_size,
                                  h.seed)
    dataset_valid = SpeLabDataset(path_dir_list / "wav_valid.txt",
                                  path_dir_list / "lab_valid.txt",
                                  path_dir_param / "norm.json",
                                  path_dir_param / "phoneme.json",
                                  hd.n_fft,
                                  hd.win_size,
                                  hd.hop_size,
                                  hd.num_mels,
                                  hd.sampling_rate,
                                  hd.fmin,
                                  hd.fmax,
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

    num_class = get_num_class(path_dir_param / "phoneme.json")
    recognizer = Conformer(num_class)
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
        best_cer = cp["best_cer"]
        del cp
        print(f"loaded {path_cp}")
    else:
        epoch = 0
        step = 0
        best_cer = 1.0

    # prepare loss function

    f_nllloss = torch.nn.NLLLoss()
    f_cer = CER(blank=-1)

    # prepare tensorboard
    sw = SummaryWriter(path_dir_log)

    # start training

    while(epoch <= h.epochs):
        epoch += 1
        print(f"--- epoch {epoch} train ---")

        # train

        for batch in tqdm(dataloader_train):
            step += 1

            # forward

            spe, lab = [item.to(h.device) for item in batch]

            prob_lab_h = recognizer(spe)

            loss = f_nllloss(prob_lab_h.transpose(-1, -2), lab)

            # backward

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # write log
            sw.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            scheduler.step()

            sw.add_scalar("train/NLL_loss", loss, step)

        # valid

        if epoch % h.valid_interval == 0:
            print(f"--- epoch {epoch} valid ---")

            recognizer.eval()
            with torch.no_grad():
                loss = 0.0
                cer = 0.0
                for batch in tqdm(dataloader_valid):
                    # forward

                    spe, lab = [item.to(h.device) for item in batch]

                    prob_lab_h = recognizer(spe)

                    loss += f_nllloss(prob_lab_h.transpose(-1, -2), lab)
                    cer += f_cer(prob_lab_h.to("cpu"), lab.to("cpu"))
                loss /= len(dataloader_valid)
                cer /= len(dataloader_valid)

                # write log
                sw.add_scalar("valid/NLL_loss", loss, step)
                sw.add_scalar("valid/CER", cer, step)
                print(f"NLL loss: {loss}, CER: {cer}")

                # save state

                torch.save(
                    recognizer.state_dict(),
                    path_dir_checkpoint / "recognizer_latest.cp")
                torch.save(
                    {"optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(),
                     "epoch": epoch,
                     "step": step,
                     "best_cer": best_cer},
                    path_dir_checkpoint / "state_latest.cp")

                print("saved recognizer_latest.cp and state_latest.cp")

                if cer <= best_cer:
                    best_cer = cer
                    torch.save(
                        recognizer.state_dict(),
                        path_dir_checkpoint / "recognizer_best.cp")
                    print("saved recognizer_best.cp")

            recognizer.train()


if __name__ == "__main__":
    train()
