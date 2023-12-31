import copy
import sys
sys.path.append('../../Nina repo')
import scipy.optimize._minimize
import torchvision
from dataloader.dataloader import DISEASE_LABELS_NIH,NIHDataResampleModule, DISEASE_LABELS_CHE, CheXpertDataResampleModule
from prediction.models import ResNet,DenseNet
import torchvision.transforms as T


import os
import torch


import matplotlib.pyplot as plt
plt.style.use("ggplot")

from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from misc import *
from gmm_and_mix_class import *
from eval_and_plotting import eval_and_plotting


def main(args, female_perc_in_training=None, random_state=None, chose_disease_str=None,
         outliers_train=[], invert_outliers_train=False, outliers_val=[], invert_outliers_val=False, use_AMP=True,
         method = "plain weighted", test_mode=False):

   # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
   #pl.seed_everything(42, workers=True)

   use_cuda = torch.cuda.is_available()
   device = torch.device("cuda:" + str(args.dev) if use_cuda else "cpu")
   print('DEVICE:{}'.format(device))

   # get run_config
   run_config = f'{args.dataset}-{chose_disease_str}' # dataset and the predicted label
   run_config+= f'-fp{female_perc_in_training}-npp{args.npp}-rs{random_state}' #f_per, npp and rs

   # if the hp value is not default
   # args_dict = vars(args)
   # for each_hp in hp_default_value.keys():
   #     if (hp_default_value[each_hp] != args_dict[each_hp] and
   #             each_hp!="num_workers"):
   #
   #         run_config+= f'-{each_hp}{args_dict[each_hp]}'

   # print('------------------------------------------\n'*3)
   # print('run_config: {}'.format(run_config))

   # Create output directory
   # out_name = str(model.model_name)
   run_dir = args.run_dir#'/work3/ninwe/run/cause_bias/'
   out_dir = run_dir + run_config
   if not os.path.exists(out_dir):
      os.makedirs(out_dir)

   cur_version = get_cur_version(out_dir)

   if args.dataset == 'NIH':
      data = NIHDataResampleModule(img_data_dir=args.img_data_dir,
                                   csv_file_img=args.csv_file_img,
                                   image_size=args.image_size,
                                   pseudo_rgb=False,
                                   batch_size=args.bs, #90 is limit i.e. 10.9gb vram
                                   num_workers=args.num_workers,
                                   augmentation=args.augmentation,
                                   outdir=out_dir,
                                   version_no=cur_version,
                                   female_perc_in_training=female_perc_in_training,
                                   chose_disease=chose_disease_str,
                                   random_state=random_state,
                                   num_classes=args.num_classes,
                                   num_per_patient=args.npp,
                                   crop=args.crop,
                                   prevalence_setting = args.prevalence_setting,

                                   )
   elif args.dataset == 'chexpert':
      if args.crop != None:
         raise Exception('Crop experiment not implemented for chexpert.')
      data = CheXpertDataResampleModule(img_data_dir=args.img_data_dir,
                                        csv_file_img=args.csv_file_img,
                                        image_size=args.image_size,
                                        pseudo_rgb=False,
                                        batch_size=args.bs, #90 is limit i.e. 10.9gb vram
                                        num_workers=args.num_workers,
                                        augmentation=args.augmentation,
                                        outdir=out_dir,
                                        version_no=cur_version,
                                        female_perc_in_training=female_perc_in_training,
                                        chose_disease=chose_disease_str,
                                        random_state=random_state,
                                        num_classes=args.num_classes,
                                        num_per_patient=args.npp,
                                        prevalence_setting = args.prevalence_setting

                                        )

   else:
      raise Exception('not implemented')

   # model
   if args.model == 'resnet':
      model_type = ResNet
   elif args.model == 'densenet':
      model_type = DenseNet
   model = model_type(num_classes=args.num_classes,lr=args.lr,pretrained=args.pretrained,model_scale=args.model_scale,
                      loss_func_type = 'BCE')

   model.to(device)
   class_imbalance_train = np.delete(data.df_train[chose_disease_str].values, outliers_train).mean() # 0.0374
   class_imbalance_init = 9.0
   loss_fn = bce_with_logits_mannual(class_imb=class_imbalance_init, n=len(data.df_train))
   #optimizer = torch.optim.Adam(params=model.model.parameters(), lr=lr)
   optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, foreach=True, fused=False) # 2.02
   #scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10)
   scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10)
   #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)
   AMP_scaler = torch.cuda.amp.GradScaler()

   exp_name = f"LAST_exp_{method}"
   if outliers_train: exp_name += f"_outliersTrain"
   if outliers_val: exp_name += f"_outliersVal"
   path = os.path.join(os.getcwd(), exp_name)
   os.makedirs(path, exist_ok=True)

   L = {"lrs_to_plot":[args.lr],
        "losses" : [],
        "accs" : [],
        "accu_female_ALL" : [0.5],
        "accu_male_ALL" : [0.5],
        "auroc_male_ALL" : [0.5],
        "auroc_female_ALL" : [0.5],
        "num_samples" : [0],
        "best_avg_auroc" : [0,0],}


   augment = T.Compose([
      T.RandomHorizontalFlip(p=0.5),
      T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
   ])

   outliers_train = torch.tensor(outliers_train, device=device, dtype=torch.int)

   #from torch.utils.data import ConcatDataset
   #data.train_set = ConcatDataset([data.train_set, data.test_set])
   data_loader = data.train_dataloader()

   beta = torch.distributions.beta.Beta(20.0, 20.0)
   print(f"\nusing {method} approch/method")
   assert method in ["label smoothing" ,"plain BCE", "BCE weighted", "gmm and mixup", "mixup", "gmm"]

   import msvcrt
   for epoch in range(args.epochs):
      if msvcrt.kbhit():
         if str(msvcrt.getch()) == ("b'\\x11'"):  # control+q to break and run one last epoch
            print("control+q detected, breaking training loop")
            break

      model.train()

      prog_bar = tqdm(data_loader, unit="batches")
      prog_bar.set_description(f"train epoch {epoch+1}/{args.epochs}")
      prog_bar.set_postfix({"male val acc":L["accu_male_ALL"][-1], "female val acc": L["accu_female_ALL"][-1] })

      losses_non_reduced = []
      ids = []
      predicted_class_train = []
      ys = []
      for i, xy in enumerate(prog_bar):
         if test_mode and i == 2:
            break

         y = xy["label"].to(device)
         x = xy["image"].to(device)
         id = xy["id"].to(device)

         with torch.no_grad():
            has_correct_label = torch.isin(id, outliers_train, invert=True).squeeze()
            if invert_outliers_train:
               y[~has_correct_label] = (y[~has_correct_label].to(torch.int)^1).to(torch.float) #inverts: 1->0 and 0->1.
            else: # remove incorrect
               y = y[has_correct_label]
               x = x[has_correct_label]
               id = id[has_correct_label]

         x = augment(x)
         ys.append(y)
         ids.append(id)

         with torch.amp.autocast(enabled=use_AMP, device_type="cuda", dtype=torch.float16):
            if method == "label smoothing":
               logits_pure = model.forward(x)
               yss = torch.cat([1 - y, y], dim=1)
               probs = torch.cat([1. - torch.sigmoid(logits_pure), torch.sigmoid(logits_pure)], dim=1)
               loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)(probs, yss)
               #loss = loss_fn.BCE_torch_plain(logits_pure, torch.clip(y, 0.1, 0.9))

            elif method == "plain BCE":
               logits_pure = model.forward(x)
               loss = loss_fn.bce_with_logits_plain(logits = logits_pure, y=y, reduce=True)

            elif method == "BCE weighted":
               logits_pure = model.forward(x)
               loss = loss_fn.bce_with_logits_plain(logits = logits_pure, y=y, s=loss_fn.class_imb_scaling, reduce=True)

            elif method == "gmm and mixup":
               x_mix, y_mix, delta, perm = mixup(x, y, beta, device)
               logit_mixup = model.forward(x_mix)

               with torch.no_grad():
                  logits_pure = model.forward(x)
                  loss_pure = loss_fn.bce_with_logits_plain(logit_mixup, y)

               loss1 = loss_fn.bce_mix_up_gmm(logit_mixup, loss_pure, y_mix, perm=None, epoch=epoch)
               loss2 = loss_fn.bce_mix_up_gmm(logit_mixup, loss_pure, y_mix, perm=perm, epoch=epoch)
               loss = 0.5*(loss1 + loss2)

            elif method == "mixup":
               x_mix, y_mix, delta, perm = mixup(x, y, beta, device)
               logit_mixup = model.forward(x_mix)
               loss = loss_fn.bce_with_logits_plain(logit_mixup, y_mix, s=loss_fn.class_imb_scaling, reduce=True)
               with torch.no_grad():
                  logits_pure = model.forward(x)

            elif method == "gmm":
               logits_pure = model.forward(x)
               with torch.no_grad():
                  loss_pure = loss_fn.bce_with_logits_plain(logits_pure, y)
               loss = loss_fn.bce_mix_up_gmm(logits_pure, loss_pure, y, perm=None, epoch=epoch)

            else:
               AssertionError("pick a proper model!")


         optimizer.zero_grad()

         if use_AMP:
            AMP_scaler.scale(loss).backward()
            AMP_scaler.step(optimizer)
            AMP_scaler.update()
         else:
            loss.backward()
            optimizer.step()

         logits_pure = logits_pure.detach()
         losses_non_reduced.append(loss_fn.bce_with_logits_plain(logits_pure, y))
         prediction = torch.sigmoid(logits_pure)
         predicted_class_train.append(prediction > 0.5)
         correct_prediction = ((prediction > 0.5) == y).to(torch.float)
         L["accs"].append(torch.mean(correct_prediction))

         L["losses"].append(loss.detach())

      predicted_class_train = torch.concat(predicted_class_train, dim=0)
      avg_predicted_class_train = predicted_class_train.to(torch.float).mean().item()
      loss_fn.class_imb_scaling_step(0.5, avg_predicted_class_train, class_imbalance_train)
      #

      losses_non_reduced = torch.concat(losses_non_reduced)
      ids = torch.concat(ids)
      if "gmm" in method:
         if args.dataset == "NIH":
            is_female_train = data.df_train["Patient Gender"].values == "F"
            #if not invert_outliers_train: is_female_train = np.delete(is_female_train, outliers_train.cpu().tolist())
         else:
            is_female_train = data.df_train["sex"].values == "Female"
            #if not is_female_train: is_female_train = np.delete(is_female_train, outliers_train.cpu().tolist())

         loss_fn.fit_and_plot_gmm(epoch_num=epoch,
                                  losses_non_reduced=losses_non_reduced,
                                  y=torch.concat(ys).cpu().numpy().squeeze(),
                                  preds=predicted_class_train.to(torch.int).cpu().numpy().squeeze(),
                                  is_female=is_female_train,
                                  ids = ids,
                                  plot_hist=True)

      L["lrs_to_plot"].append(scheduler.get_last_lr()[0])
      L["num_samples"].append((epoch + 1) * len(data_loader))
      scheduler.step()

      eval_and_plotting(L=L, model=model, device=device, data_loader=data.val_dataloader(), data_set=data.df_valid,
                        data_set_name=args.dataset, outliers_list=outliers_val, invert_outliers=invert_outliers_val,
                        epoch=epoch, avg_predicted_class_train=avg_predicted_class_train,
                        class_imb_scaling=loss_fn.class_imb_scaling, method=method, model_scale=args.model_scale,
                        batch_size=args.bs, name="", path=path)

      if L["best_avg_auroc"][0] < 0.5*L["auroc_female_ALL"][-1] + 0.5*L["auroc_male_ALL"][-1] and epoch >= 1:
         L["best_avg_auroc"][0] = 0.5*L["auroc_female_ALL"][-1] + 0.5*L["auroc_male_ALL"][-1]
         L["best_avg_auroc"][1] = L["num_samples"][-1]
         new_best = L["best_avg_auroc"][0]
         print(f"found new best: with auroc {new_best:.4f}" )
         print("Running on test set!")
         torch.save(model, os.path.join(path, f"model.pt"))
         temp_L = make_temp_L(L) # appending the fake stuff
         eval_and_plotting(L=temp_L, model=model, device=device, data_loader=data.test_dataloader(),
                           data_set=data.df_test,
                           data_set_name=args.dataset, outliers_list=[], invert_outliers=False,
                           epoch=epoch, avg_predicted_class_train=avg_predicted_class_train,
                           class_imb_scaling=loss_fn.class_imb_scaling, method=method, model_scale=args.model_scale,
                           batch_size=args.bs, name="BEST TEST ", path=path)
         print("BEST TEST AUROC: male", temp_L["auroc_male_ALL"][-1], "female", temp_L["auroc_female_ALL"][-1])


   print("\nRunning on test set!")
   temp_L = make_temp_L(L) # appending the fake stuff
   eval_and_plotting(L=temp_L, model=model, device=device, data_loader=data.test_dataloader(), data_set=data.df_test,
                     data_set_name=args.dataset, outliers_list=[], invert_outliers=False,
                     epoch=epoch, avg_predicted_class_train=avg_predicted_class_train,
                     class_imb_scaling=loss_fn.class_imb_scaling, method=method, model_scale=args.model_scale,
                     batch_size=args.bs, name="LAST TEST ", path=path)
   print("FINAL TEST AUROC: male", temp_L["auroc_male_ALL"][-1], "female", temp_L["auroc_female_ALL"][-1])

if __name__ == '__main__':
    parser = ArgumentParser()
    #parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)

    #dataset_default = "chexpert"
    #img_dir_default = r"C:\Users\Karlu\Desktop\11\Learning From Noisy Data\archivepreproc_224x224"

    dataset_default = "NIH"
    img_dir_default = r"C:\Users\Karlu\Desktop\11\Learning From Noisy Data\NIH\preproc_224x224"


    disease_label_default = ['Pneumothorax']
    female_percent_in_training_default = "50"
    npp_default = 1 #Number per patient, could be integer or None (no sampling)
    run_dir_default = r"C:\Users\Karlu\Desktop\11\Learning From Noisy Data\runs"

    #shutdown -s -f -t 6000
    epochs_default = 35
    pretrained_default = True
    save_model_default = False
    num_workers_default = 0
    augmentation_default = True
    model_scale_default = "50"# "50" #'convnext' # "efficientformer"
    batch_size_default = 65
    lr_default = 2e-5
    method = "" #["label smoothing" ,"plain BCE", "BCE weighted", "gmm and mixup", "mixup", "gmm"]
    use_AMP = False
    test_mode = False
    from CL_FOUND_OUTLIERS import (OUTLIER_IDS_MIXUP_1PERCENT, OUTLIER_IDS_MIXUP_2PERCENT,
                                   VAL_OUTLIER_IDS_MIX_1PERCENT, VAL_OUTLIER_IDS_MIX_2PERCENT)

    outliers_train = OUTLIER_IDS_MIXUP_1PERCENT # OUTLIER_IDS_PLAIN_1PERCENT, OUTLIER_IDS_MIXUP_1PERCENT
    invert_outliers_train = False

    outliers_val = VAL_OUTLIER_IDS_MIX_1PERCENT #VAL_OUTLIER_IDS_MIX_2PERCENT#[] # VAL_OUTLIER_IDS_PLAIN_2PERCENT, VAL_OUTLIER_IDS_MIX_2PERCENT
    invert_outliers_val = False






    # hps that need to chose when training
    parser.add_argument('-s','--dataset',default=dataset_default,help='Dataset', choices =['NIH','chexpert'])
    parser.add_argument('-d','--disease_label',default=disease_label_default, help='Chosen disease label', type=str, nargs='*')
    parser.add_argument('-f', '--female_percent_in_training', default=female_percent_in_training_default,
                        help='Female percentage in training set, should be any of [0, 50, 100]', type=str, nargs='+')
    parser.add_argument('-n', '--npp',default=npp_default, help='Number per patient, could be integer or None (no sampling)',type=int)
    parser.add_argument('-r', '--random_state', default='0-10', help='random state')
    parser.add_argument('-p','--img_dir', default=img_dir_default, help='your img dir path here',type=str)
    parser.add_argument('-rd','--run_dir', default=run_dir_default, help='where the runs are saved',type=str)

    # hps that set as defaults
    parser.add_argument('--lr', default=lr_default, help='learning rate, default=1e-6')
    parser.add_argument('--bs', default=batch_size_default, help='batch size, default=64')
    parser.add_argument('--epochs',default=epochs_default,help='number of epochs, default=20')
    parser.add_argument('--model', default='resnet', help='model, default=\'ResNet\'')
    parser.add_argument('--model_scale', default=model_scale_default, help='model scale, default=50',type=str)
    parser.add_argument('--pretrained', default=pretrained_default, help='pretrained or not, True or False, default=True',type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--augmentation', default=augmentation_default, help='augmentation during training or not, True or False, default=True',type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--is_multilabel',default=False,help='training with multilabel or not, default=False, single label training',type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--image_size', default=224,help='image size',type=int)
    parser.add_argument('--crop',default=None,help='crop the bottom part of the image, the percentage of cropped part, when cropping, default=0.6')
    parser.add_argument('--prevalence_setting',default='separate',help='which kind of prevalence are being used when spliting,\
                        choose from [separate, equal, total]',choices=['separate','equal','total'])
    parser.add_argument('--save_model',default=save_model_default,help='dave model parameter or not',type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--num_workers', default=num_workers_default, help='number of workers')

    args = parser.parse_args()


    # other hps
    if args.is_multilabel:
        args.num_classes = len(DISEASE_LABELS_NIH) if args.dataset == 'NIH' else len(DISEASE_LABELS_CHE)
    else: args.num_classes = 1


    if args.image_size == 224:
        args.img_data_dir = args.img_dir+'{}/preproc_224x224/'.format(args.dataset)
    elif args.image_size == 1024:
        args.img_data_dir = args.img_dir+'{}/images/'.format(args.dataset)

    if args.dataset == 'NIH':
        args.csv_file_img = '../datafiles/'+'Data_Entry_2017_v2020_clean_split.csv'
    elif args.dataset == 'chexpert':
        args.csv_file_img = '../datafiles/'+'chexpert.sample.allrace.csv'
    else:
        raise Exception('Not implemented.')

    #print('hyper-parameters:')
    #print(args)

    if len(args.random_state.split('-')) != 2:
        if len(args.random_state.split('-')) == 1:
            rs_min, rs_max = int(args.random_state), int(args.random_state)+1
        else:
            raise Exception('Something wrong with args.random_states : {}'.format(args.random_states))
    rs_min, rs_max = int(args.random_state.split('-')[0]),int(args.random_state.split('-')[1])

    female_percent_in_training_set = [int(percent) for percent in args.female_percent_in_training.split(" ")]
    print('female_percent_in_training_set:{}'.format(female_percent_in_training_set))
    disease_label_list = args.disease_label #[''.join(each) for each in args.disease_label]
    if len(disease_label_list) ==1 and disease_label_list[0] == 'all':
        disease_label_list = DISEASE_LABELS_NIH if args.dataset == 'NIH' else DISEASE_LABELS_CHE
    print('disease_label_list:{}'.format(disease_label_list))


    print('***********RESAMPLING EXPERIMENT**********\n')
    for d in disease_label_list:
        for female_perc_in_training in female_percent_in_training_set:
            for i in np.arange(rs_min, rs_max):
               for method in ["BCE weighted", "gmm and mixup", "mixup", "gmm"]:
                   main(args, female_perc_in_training=female_perc_in_training,random_state = i,chose_disease_str=d,
                        method=method, outliers_train=outliers_train, invert_outliers_train=invert_outliers_train,
                        outliers_val=outliers_val, invert_outliers_val=invert_outliers_val, use_AMP=use_AMP,
                        test_mode=test_mode)