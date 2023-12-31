import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from scipy.stats import norm

@torch.no_grad()
def mixup(x, y, beta_dist, d):
    n_samples = x.shape[0]
    delta = beta_dist.sample(sample_shape=(n_samples, 1, 1, 1)).to(d)
    perm = torch.randperm(n_samples, device=d)
    x_mix = delta * x + (1 - delta) * x[perm]
    delta = delta.squeeze()[:, None]
    y_mix = delta * y + (1 - delta) * y[perm]

    return x_mix, y_mix, delta, perm

class bce_with_logits_mannual(torch.nn.Module):
    def __init__(self, class_imb=6.0, n=100_000, eps=1e-6, device="cuda"):
        super().__init__()
        self.class_imb_scaling = class_imb
        self.eps = eps
        self.n = n
        self.d = device
        self.high_loss_occourence = np.zeros(self.n)
        self.BCE_torch_plain = torch.nn.BCEWithLogitsLoss()
        # self.z  = None

    def bce_with_logits_plain(self, logits, y, s=1.0, reduce=False):
        h = torch.nn.functional.sigmoid(logits)
        h = torch.clamp(h, self.eps, 1.0 - self.eps)

        loss = -(s * y * torch.log(h) + (1.0 - y) * torch.log(1.0 - h))

        if reduce:
            return loss.mean()

        return loss


    def bce_mix_up_gmm(self, logits_mixup, loss, y, perm, epoch=0):
        if epoch == 0:
            self.w = torch.ones(logits_mixup.shape[0], dtype=float, device=self.d)

        else:
            if perm is None:
                loss_normed = loss / self.loss_norm_factor
                self.w = self.gmm_fit.predict_proba(loss_normed.cpu().detach().numpy())[:, 0]
                self.w = torch.from_numpy(self.w).to(float).to(self.d)
                if not self.gmm_means_ordered:
                    self.w = 1.0 - self.w
            else:
                self.w = self.w[perm]

        h = torch.nn.functional.sigmoid(logits_mixup)
        h = torch.clamp(h, self.eps, 1 - self.eps)

        # z = (h > 0.5).to(torch.float) # hard dynamic boot strapping
        z1 = torch.clone(h)  # soft dynamic boot strapping

        w_l = self.w # prob that the loss is low. i.e how much we trust the label
        w_h = 1.0 - self.w # prob that the loss is high i.e. how much we dont trust the label
        y1 = y
        y0 = 1.0-y
        s = self.class_imb_scaling
        z0 = 1.0 - z1

        # if we dont trust the label (i.e. when w_l approches 0) we will use the prediction as the label
        class_1 = (w_l * y1 + w_h * z1) * torch.log(h) * s
        class_0 = (w_l * y0 + w_h * z0) * torch.log(1.0 - h)
        #class_0 = (w_l * y0 + w_h * z1) * torch.log(1.0 - h)

        return (-(class_1 + class_0)).mean()

    def fit_and_plot_gmm(self,
                        epoch_num,
                        losses_non_reduced=None,
                        ids=None,
                        preds=None,
                        y=None,
                        is_female=None,
                        plot_hist=False):

        # losses = torch.zeros((self.n, 1), device=self.d, dtype=torch.float)
        # losses[ids] = losses_non_reduced
        losses = losses_non_reduced

        losses = losses.cpu().numpy()
        self.loss_norm_factor = losses.max()
        losses /= self.loss_norm_factor
        self.gmm_fit = GaussianMixture(n_components=2, means_init=np.array([0.0, 1.0])[:, None]).fit(losses)
        losses_scored = self.gmm_fit.predict_proba(losses)

        gmm_means_ordered = (self.gmm_fit.means_[0] < self.gmm_fit.means_[1])[0]
        self.gmm_means_ordered = gmm_means_ordered
        if gmm_means_ordered:
            low_idx, high_idx = 0, 1
        else:
            low_idx, high_idx = 1, 0

        # self.w_low_loss = torch.from_numpy(losses_scored[:, low_idx]).to(torch.float).to(self.d)
        # self.w_high_loss = torch.from_numpy(losses_scored[:, high_idx]).to(torch.float).to(self.d)

        if plot_hist:
            losses_pred = self.gmm_fit.predict(losses)
            args = [{"c": "cornflowerblue", "title": "low loss count: "},
                    {"c": "indianred", "title": "high loss count: "}]
            fig, ax = plt.subplots(2, 2)
            fig.suptitle(f"fitted gmms epoch {epoch_num}")
            plt.tight_layout()
            fig.set_figheight(10)
            fig.set_figwidth(10)

            for idx in [low_idx, high_idx]:
                loss_count = str((losses_pred == idx).sum())
                losses_ = losses.squeeze()[losses_pred == idx]
                x_axis_range = np.linspace(0, losses_.max() * 1.2, 1000)
                pdf = norm.pdf(x_axis_range, self.gmm_fit.means_[idx, 0], np.sqrt(self.gmm_fit.covariances_[idx, 0]))

                ax[0, idx].hist(losses_, label=args[idx]["title"] + loss_count, bins=30, density=True,
                                color=args[idx]["c"])
                ax[0, idx].plot(x_axis_range, pdf, color="black", ls="--", label="fitted gmm")
                ax[0, idx].legend()

            self.high_loss_occourence[ids.cpu().numpy()] += (losses_pred == high_idx)
            top_10 = np.flip(np.argsort(self.high_loss_occourence))[:10]
            print("top idx", top_10)
            print("top count", self.high_loss_occourence[top_10])

            high_idx = losses_pred == high_idx
            is_female = is_female[ids.cpu().numpy()]

            high_ids_male = high_idx * ~is_female
            high_ids_female = high_idx * is_female

            try:
                ConfusionMatrixDisplay.from_predictions(y[high_ids_male], preds[high_ids_male], ax=ax[1, 0])
                ax[1, 0].set_title(f"MALE high loss (tot {sum(high_ids_male)})")
                ConfusionMatrixDisplay.from_predictions(y[high_ids_female], preds[high_ids_female], ax=ax[1, 1])
                ax[1, 1].set_title(f"FEMALE high loss (tot {sum(high_ids_female)})")
            except Exception:
                pass
            plt.show()

    def forward(self, logits, y, id, epoch_num):
        return self.bce_with_logits_weighted(logits, y, id, epoch_num)

    def class_imb_scaling_step(self,
                               rate=0.5,
                               avg_predicted_class_train=None,
                               class_imbalance_train=None, multiply=False):

        if multiply:
            self.class_imb_scaling *= (class_imbalance_train/avg_predicted_class_train)
            self.class_imb_scaling = np.clip(self.class_imb_scaling, 1.0, self.class_imb_scaling*3)
        else:
            self.class_imb_scaling += (class_imbalance_train - avg_predicted_class_train)*rate*100.0
            self.class_imb_scaling = np.clip(self.class_imb_scaling, 1.0, 20)

        # max_ = 1 / class_imbalance_train
        # if avg_predicted_class_train < class_imbalance_train:
        #     self.class_imb_scaling += rate
        # else:
        #     self.class_imb_scaling -= rate
        #
        # self.class_imb_scaling = np.clip(self.class_imb_scaling, 0.1, max_)