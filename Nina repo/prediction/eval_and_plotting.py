import matplotlib.pyplot as plt
import torch
from misc import ewma
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import os


@torch.no_grad()
def eval_and_plotting(L, model, device, data_loader, data_set, data_set_name, outliers_list, invert_outliers, epoch,
                      avg_predicted_class_train, class_imb_scaling, method, model_scale, batch_size, name="", path=None):


   if data_set_name == "NIH":
      is_female_val = data_set["Patient Gender"].values == "F"
      if not invert_outliers: is_female_val = np.delete(is_female_val, outliers_list)
      is_female_val = torch.from_numpy(is_female_val).to(torch.bool).to(device)
   else:
      is_female_val = data_set["sex"].values == "Female"
      if not invert_outliers: is_female_val = np.delete(is_female_val, outliers_list)
      is_female_val = torch.from_numpy(is_female_val).to(torch.bool).to(device)

   outliers_list = torch.tensor(outliers_list, device=device, dtype=torch.int)

   model.eval()
   pred_probs = []
   labels = []

   for xy in data_loader:
      x = xy["image"].to(device)
      y = xy["label"].to(device)
      id = xy["id"].to(device)

      has_correct_label = torch.isin(id, outliers_list, invert=True).squeeze()

      if invert_outliers:
         y[~has_correct_label] = (y[~has_correct_label].to(torch.int) ^ 1).to(torch.float)
      else:  # else remove
         x = x[has_correct_label]
         y = y[has_correct_label]

      logit = model.forward(x)
      pred_prob = torch.sigmoid(logit)

      pred_probs.append(pred_prob)
      labels.append(y)

   last_lr = L["lrs_to_plot"][-1]
   avg_pred_val = torch.mean((torch.concat(pred_probs, dim=0) > 0.5).to(torch.float)).item()
   print(f"E{epoch + 1}: "
         f" avg val pred class:"
         f" {avg_pred_val:.4f},"
         f" avg train pred class:"
         f" {avg_predicted_class_train:.4f},"
         f" loss_fn.class_imb_scaling: {class_imb_scaling:.1f}"
         f" curr lr: {last_lr:.7f}")

   labels = torch.concat(labels, dim=0)
   pred_probs = torch.concat(pred_probs, dim=0)

   predictions_female = pred_probs[is_female_val]
   labels_female = labels[is_female_val]
   accu_female = model.accu_func(predictions_female, labels_female).item()
   auroc_female = model.auroc_func(predictions_female, labels_female).item()


   predictions_male = pred_probs[~is_female_val]
   labels_male = labels[~is_female_val]
   accu_male = model.accu_func(predictions_male, labels_male).item()
   auroc_male = model.auroc_func(predictions_male, labels_male).item()


   L["accu_female_ALL"].append(accu_female)
   L["auroc_female_ALL"].append(auroc_female)
   L["accu_male_ALL"].append(accu_male)
   L["auroc_male_ALL"].append(auroc_male)

   # prog_bar.set_postfix({"cur lr": scheduler.get_last_lr()})
   plt.style.use("ggplot")
   fig = plt.figure(figsize=(10, 7))
   # add grid specifications
   gs = fig.add_gridspec(3, 4)
   # open axes/subplots
   axs = []
   axs.append(fig.add_subplot(gs[:, 0:3]))  # large subplot (2 rows, 2 columns)
   # axs.append(fig.add_subplot(gs[0, 3]))  # small subplot (1st row, 3rd column)
   # axs.append(fig.add_subplot(gs[1, 3]))
   # axs.append(fig.add_subplot(gs[2, 3]))
   accu_male_ALL, accu_female_ALL = L["accu_male_ALL"][-1], L["accu_female_ALL"][-1]
   auroc_male_ALL, auroc_female_ALL = L["auroc_male_ALL"][-1], L["auroc_female_ALL"][-1]

   plt.plot(ewma(torch.stack(L["losses"]).cpu().numpy().astype(np.float64), 50) / 2. + 0.5, label="TRAIN loss (smooth)",color="seagreen",alpha=0.8)
   plt.plot(np.clip(ewma(torch.stack(L["accs"]).cpu().numpy().astype(np.float64), 50), 0.5, 1.0), label="TRAIN accu (smooth)", ls="--", alpha=0.5, color="seagreen")

   plt.plot(L["num_samples"], L["accu_male_ALL"], marker='o', label=f"val acc male: {accu_male_ALL:.3f}", color="cornflowerblue", ls="--", alpha=0.5)
   plt.plot(L["num_samples"], L["accu_female_ALL"], marker='o', label=f"val acc female: {accu_female_ALL:.3f}",color="indianred", ls="--", alpha=0.5)

   plt.plot(L["num_samples"], L["auroc_male_ALL"], marker='*', label=f"val AUROC male: {auroc_male_ALL:.3f}", color="cornflowerblue")
   plt.plot(L["num_samples"], L["auroc_female_ALL"], marker='*', label=f"val AUROC female: {auroc_female_ALL:.3f}", color="indianred")

   if L["best_avg_auroc"][0] != 0:
      plt.axvline(L["best_avg_auroc"][1], ls="--", alpha=0.4, color="gray", label="best val AUROC")#("+L["best_avg_auroc"][0])

   # plt.plot(L["num_samples"], np.asarray(L["lrs_to_plot"]) / np.asarray(L["lrs_to_plot"]).max() / 2 + 0.5,
   #          label=f"LR (normalized {last_lr:.5f})",
   #          marker='o', alpha=0.5)
   plt.title(f"{name}{data_set_name}: {method}, resnet{model_scale}, epoch: {epoch + 1}")
   batch_num_array = np.linspace(0, L["num_samples"][-1], 5, dtype=int)
   plt.xticks(batch_num_array, batch_num_array * batch_size)
   plt.xlabel("training samples")
   # plt.ylabel("loss and accu")
   plt.legend(loc='lower left')

   plt.style.use("default")
   axs.append(fig.add_subplot(gs[0, 3]))  # small subplot (1st row, 3rd column)
   axs.append(fig.add_subplot(gs[1, 3]))
   axs.append(fig.add_subplot(gs[2, 3]))

   val_pred = (pred_probs > 0.5).to(torch.int)
   norm = "all"
   ConfusionMatrixDisplay.from_predictions(labels[~is_female_val].cpu().to(torch.int).numpy(),
                                           val_pred[~is_female_val].cpu().to(torch.int).numpy(), ax=axs[1], colorbar=False,
                                           normalize=norm, cmap="cividis")
   axs[1].set_title(f"Male")
   axs[1].axis("off")

   ConfusionMatrixDisplay.from_predictions(labels[is_female_val].cpu().to(torch.int).numpy(),
                                           val_pred[is_female_val].cpu().to(torch.int).numpy(), ax=axs[2], colorbar=False,
                                           normalize=norm, cmap="cividis")
   axs[2].set_title(f"Female")
   axs[2].axis("off")

   ConfusionMatrixDisplay.from_predictions(labels.cpu().to(torch.int).numpy(),
                                           val_pred.cpu().to(torch.int).numpy(), ax=axs[3], colorbar=False,
                                           normalize=norm, cmap="cividis")
   axs[3].set_title(f"Combined")
   # axs[3].axis("off")
   # axs[3].set_label("0  (pred)   1")




   plt.tight_layout()
   # if epoch > 49:
   #    plt.savefig(f"plainplain{epoch}.png")
   plt.savefig(os.path.join(path, f"{name}_plot.png"), dpi=150)
   if (epoch+1) % 7 == 0:
      plt.show()