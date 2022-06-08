import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from pathlib import Path
from importlib import import_module
import json
from attrdict import AttrDict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.multiprocessing as multiprocessing
from torch.utils.tensorboard import SummaryWriter

import sys
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[1]))
    sys.path.append(str(Path(__file__).parents[2]))

from train.tool.dataset import SpeDataset
from train.tool.standardizer import Standardizer
from train.tool.losses import f_loss_adv, f_reg_r1, f_loss_f0
from common.model.models import Generator, StyleEncoder, Discriminator, Classifier

torch.backends.cudnn.benchmark = True


def train(path_dir_list=Path("../dataset/list"),
          path_dir_param=Path("../dataset/param"),
          path_dir_checkpoint=Path("./checkpoint"),
          path_dir_log=Path("./log"),
          path_config_data="../../HiFiGAN/config_v1.json",
          path_config_train="./config.json",
          package_JDCNet="JDCNet",
          package_CNNConformer="CNNConformer_ASR"):
    # dynamic import
    JDCNet = import_module(f"{package_JDCNet}.common.model.jdcnet").JDCNet
    Conformer = import_module(f"{package_CNNConformer}.common.conformer.model").Conformer
    get_num_class = import_module(f"{package_CNNConformer}.common.tool.get_num_class").get_num_class

    path_dir_param_jdcnet = Path(f"../../{package_JDCNet}/dataset/param")
    path_cp_jdcnet = Path(f"../../{package_JDCNet}/train/checkpoint/recognizer_best.cp")
    path_dir_param_cnnconformer = Path(f"../../{package_CNNConformer}/dataset/param")
    path_cp_cnnconformer = Path(f"../../{package_CNNConformer}/train/checkpoint/recognizer_best.cp")

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

    list_path_list_train = sorted((path_dir_list / "train").glob("*.txt"))
    dataset_train = SpeDataset(list_path_list_train,
                               path_dir_param / "norm.json",
                               hd.n_fft,
                               hd.win_size,
                               hd.hop_size,
                               hd.num_mels,
                               hd.sampling_rate,
                               hd.fmin,
                               hd.fmax,
                               h.segment_size,
                               h.seed)
    dataset_valid = SpeDataset(sorted((path_dir_list / "valid").glob("*.txt")),
                               path_dir_param / "norm.json",
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

    standardizer_recognizer_f0 = Standardizer(path_dir_param / "norm.json",
                                              path_dir_param_jdcnet / "norm.json")
    standardizer_recognizer_ppg = Standardizer(path_dir_param / "norm.json",
                                               path_dir_param_cnnconformer / "norm.json")
    # prepare model

    num_domain = len(list_path_list_train)

    generator = Generator(style_dim=h.style_dim)
    style_encoder = StyleEncoder(num_domains=num_domain, style_dim=h.style_dim)
    discriminator = Discriminator(num_domains=num_domain)
    classifier = Classifier(num_domains=num_domain)

    generator, style_encoder, discriminator, classifier = [item.to(h.device) for item in [generator, style_encoder, discriminator, classifier]]

    path_cp = path_dir_checkpoint / "generator_latest.cp"
    if path_cp.exists():
        cp = torch.load(path_cp, map_location=lambda storage, loc: storage)
        generator.load_state_dict(cp)
        del cp
        print(f"loaded {path_cp}")
    path_cp = path_dir_checkpoint / "style_encoder_latest.cp"
    if path_cp.exists():
        cp = torch.load(path_cp, map_location=lambda storage, loc: storage)
        style_encoder.load_state_dict(cp)
        del cp
        print(f"loaded {path_cp}")

    # prepare optimizer

    optimizer_generator = torch.optim.RAdam(generator.parameters(),
                                            h.lr_generator,
                                            h.betas,
                                            h.eps,
                                            h.weight_decay)
    optimizer_style_encoder = torch.optim.RAdam(style_encoder.parameters(),
                                                h.lr_style_encoder,
                                                h.betas,
                                                h.eps,
                                                h.weight_decay)

    optimizer_discriminator = torch.optim.RAdam(discriminator.parameters(),
                                                h.lr_discriminator,
                                                h.betas,
                                                h.eps,
                                                h.weight_decay)
    optimizer_classifier = torch.optim.RAdam(classifier.parameters(),
                                             h.lr_classifier,
                                             h.betas,
                                             h.eps,
                                             h.weight_decay)

    path_cp = path_dir_checkpoint / "state_latest.cp"
    if path_cp.exists():
        cp = torch.load(path_cp, map_location=lambda storage, loc: storage)
        discriminator.load_state_dict(cp["discriminator"])
        classifier.load_state_dict(cp["classifier"])
        optimizer_generator.load_state_dict(cp["optimizer_generator"])
        optimizer_style_encoder.load_state_dict(cp["optimizer_style_encoder"])
        optimizer_discriminator.load_state_dict(cp["optimizer_discriminator"])
        optimizer_classifier.load_state_dict(cp["optimizer_classifier"])
        epoch = cp["epoch"]
        step = cp["step"]
        del cp
        print(f"loaded {path_cp}")
    else:
        epoch = 0
        step = 0

    # prepare models for loss

    recognizer_f0 = JDCNet()

    assert path_cp_jdcnet.exists()
    cp = torch.load(path_cp_jdcnet, map_location=lambda storage, loc: storage)
    recognizer_f0.load_state_dict(cp)
    del cp
    print(f"loaded {path_cp_jdcnet}")

    num_class = get_num_class(path_dir_param_cnnconformer / "phoneme.json")
    recognizer_ppg = Conformer(num_class)

    assert path_cp_cnnconformer.exists()
    cp = torch.load(path_cp_cnnconformer, map_location=lambda storage, loc: storage)
    recognizer_ppg.load_state_dict(cp)
    del cp
    print(f"loaded {path_cp_cnnconformer}")

    recognizer_f0.eval_grad()
    recognizer_ppg.eval()
    recognizer_f0, recognizer_ppg = [item.to(h.device) for item in [recognizer_f0, recognizer_ppg]]

    # prepare loss function

    f_celoss = torch.nn.CrossEntropyLoss()

    # prepare tensorboard
    sw = SummaryWriter(path_dir_log)

    # start training

    [item.eval() for item in [generator, style_encoder]]

    while(epoch < h.epochs):
        epoch += 1
        print(f"--- epoch {epoch} train ---")

        # train

        for batch in tqdm(dataloader_train):
            step += 1

            spe_input, num_input, num_target, spe_target = [item.to(h.device) for item in batch]

            # forward discriminator

            [item.train() for item in [discriminator, classifier]]

            with torch.no_grad():
                style_target = style_encoder(spe_target, num_target)
                spe_output = generator(spe_input, style_target)

            spe_target.requires_grad_()
            reality_target = discriminator(spe_target, num_target)
            reality_output = discriminator(spe_output, num_target)

            loss_d_real = f_loss_adv(reality_target, 1)
            reg_r1_d = f_reg_r1(reality_target, spe_target)
            loss_d_fake = f_loss_adv(reality_output, 0)
            loss_d = loss_d_real + loss_d_fake + h.weight_r1reg * reg_r1_d

            # backward discriminator

            optimizer_discriminator.zero_grad(set_to_none=True)
            loss_d.backward()
            optimizer_discriminator.step()

            if epoch >= h.epoch_start_classify:
                # forward classifier

                prob_speaker_input = classifier(spe_input)
                prob_speaker_output = classifier(spe_output)

                loss_c_input = f_celoss(prob_speaker_input, num_input)
                loss_c_output = f_celoss(prob_speaker_output, num_input)
                reg_r1_c = f_reg_r1(prob_speaker_input, spe_input)
                loss_c = loss_c_input + loss_c_output + h.weight_r1reg * reg_r1_c

                # backward classifier

                optimizer_classifier.zero_grad(set_to_none=True)
                loss_c.backward()
                optimizer_classifier.step()

            # forward generator, style_encoder

            [item.eval() for item in [discriminator, classifier]]
            [item.train() for item in [generator, style_encoder]]

            style_target = style_encoder(spe_target, num_target)
            spe_output = generator(spe_input, style_target)

            reality_output = discriminator(spe_output, num_target)
            loss_g_adv = f_loss_adv(reality_output, 1)

            if epoch >= h.epoch_start_classify:
                prob_speaker_output = classifier(spe_output)
                loss_g_advcls = f_celoss(prob_speaker_output, num_target)
            else:
                loss_g_advcls = 0.0

            style_input = style_encoder(spe_input, num_input)
            spe_input_cycle = generator(spe_output, style_input)
            loss_cycle = F.smooth_l1_loss(spe_input_cycle, spe_input)

            with torch.no_grad():
                f0_input, vuv_input = recognizer_f0(standardizer_recognizer_f0(spe_input))
                vuv_input = vuv_input >= 0.5
                ppg_input = recognizer_ppg(standardizer_recognizer_ppg(spe_input))

            f0_output, vuv_output = recognizer_f0(standardizer_recognizer_f0(spe_output))
            loss_f0 = f_loss_f0(f0_output, vuv_output.detach() >= 0.5, f0_input, vuv_input)

            ppg_output = recognizer_ppg(standardizer_recognizer_ppg(spe_output))
            loss_ppg = F.smooth_l1_loss(ppg_output, ppg_input)

            loss_g = loss_g_adv + h.weight_loss_g_advcls * loss_g_advcls + h.weight_loss_cycle * loss_cycle + h.weight_loss_f0 * loss_f0 + h.weight_loss_ppg * loss_ppg

            # backward generator, style_encoder

            optimizer_style_encoder.zero_grad(set_to_none=True)
            optimizer_generator.zero_grad(set_to_none=True)
            loss_g.backward()
            optimizer_style_encoder.step()
            optimizer_generator.step()

            [item.eval() for item in [generator, style_encoder]]

            # write log

            sw.add_scalar("train/adversarial_discriminator_loss_real", loss_d_real, step)
            sw.add_scalar("train/adversarial_discriminator_loss_fake", loss_d_fake, step)
            sw.add_scalar("train/adversarial_generator_loss", loss_g_adv, step)
            sw.add_scalar("train/cycle_consistency_loss", loss_cycle, step)
            sw.add_scalar("train/f0_consistency_loss", loss_f0, step)
            sw.add_scalar("train/ppg_consistency_loss", loss_ppg, step)

        # valid

        if epoch % h.valid_interval == 0:
            print(f"--- epoch {epoch} valid ---")

            with torch.no_grad():
                loss_cycle = 0.0
                loss_f0 = 0.0
                loss_ppg = 0.0
                for batch in tqdm(dataloader_valid):
                    # forward

                    spe_input, num_input, num_target, spe_target = [item.to(h.device) for item in batch]

                    style_target = style_encoder(spe_target, num_target)
                    spe_output = generator(spe_input, style_target)

                    style_input = style_encoder(spe_input, num_input)
                    spe_input_cycle = generator(spe_output, style_input)
                    loss_cycle += F.smooth_l1_loss(spe_input_cycle, spe_input)

                    f0_input, vuv_input = recognizer_f0(standardizer_recognizer_f0(spe_input))
                    f0_output, vuv_output = recognizer_f0(standardizer_recognizer_f0(spe_output))
                    loss_f0 += f_loss_f0(f0_output, vuv_output.detach() >= 0.5, f0_input, vuv_input.detach() >= 0.5)

                    ppg_input = recognizer_ppg(standardizer_recognizer_ppg(spe_input))
                    ppg_output = recognizer_ppg(standardizer_recognizer_ppg(spe_output))
                    loss_ppg += F.smooth_l1_loss(ppg_output, ppg_input)

                loss_cycle /= len(dataloader_valid)
                loss_f0 /= len(dataloader_valid)
                loss_ppg /= len(dataloader_valid)

                # write log
                sw.add_scalar("valid/cycle_consistency_loss", loss_cycle, step)
                sw.add_scalar("valid/f0_consistency_loss", loss_f0, step)
                sw.add_scalar("valid/ppg_consistency_loss", loss_ppg, step)

                # save state

                torch.save(
                    generator.state_dict(),
                    path_dir_checkpoint / "generator_latest.cp")
                torch.save(
                    style_encoder.state_dict(),
                    path_dir_checkpoint / "style_encoder_latest.cp")
                torch.save(
                    {"discriminator": discriminator.state_dict(),
                     "classifier": classifier.state_dict(),
                     "optimizer_generator": optimizer_generator.state_dict(),
                     "optimizer_style_encoder": optimizer_style_encoder.state_dict(),
                     "optimizer_discriminator": optimizer_discriminator.state_dict(),
                     "optimizer_classifier": optimizer_classifier.state_dict(),
                     "epoch": epoch,
                     "step": step},
                    path_dir_checkpoint / "state_latest.cp")

                print("saved generator_latest.cp, style_encoder_latest, and state_latest.cp")

        if epoch % h.save_interval == 0:
            torch.save(
                generator.state_dict(),
                path_dir_checkpoint / f"generator_{str(epoch).zfill(4)}.cp")
            torch.save(
                style_encoder.state_dict(),
                path_dir_checkpoint / f"style_encoder_{str(epoch).zfill(4)}.cp")


if __name__ == "__main__":
    train()
