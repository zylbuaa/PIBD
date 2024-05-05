
import torch.nn as nn
import torch
import torch.nn.functional as F



class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


# TODO: document better and clean up
def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='sum'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        The weight on uncensored loss 
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # print("h shape", h.shape)

    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h) #hazard function
    # print("hazards shape", hazards.shape)

    S = torch.cumprod(1 - hazards, dim=1)
    # print("S.shape", S.shape, S)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # hazards[y] = hazards(1)
    # S[1] = S(1)

    # print("S_padded.shape", S_padded.shape, S_padded)


    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    # print('s_prev.s_prev', s_prev.shape, s_prev)
    # print('h_this.shape', h_this.shape, h_this)
    # print('s_this.shape', s_this.shape, s_this)

    # c = 1 means censored. Weight 0 in this case 
    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    

    # print('uncensored_loss.shape', uncensored_loss.shape)
    # print('censored_loss.shape', censored_loss.shape)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss


def loss_reg_l1(coef):
    print('[setup] L1 loss with coef={}'.format(coef))
    coef = .0 if coef is None else coef
    def func(model_params):
        if coef <= 1e-8:
            return 0.0
        else:
            return coef * sum([torch.abs(W).sum() for W in model_params])
    return func

##################################################
# General Loss for Survival Analysis Models,
# including continuous output and discrete output.
##################################################

def recon_loss(pred_t, t, e, alpha=0.0, gamma=1.0, norm='l1', cur_alpha=None):
    """Continuous Survival Model

    Reconstruction loss for pred_t and labels.
    recon_loss = l2 + l3
    if e = 0, l2 = max(0, t - pred_t)
    if e = 1, l3 = |t - pred_t|
    """
    pred_t = pred_t.squeeze()
    t = t.squeeze()
    e = e.squeeze()
    loss_obs = e * torch.abs(pred_t - t)
    loss_cen = (1 - e) * F.relu(gamma - (pred_t - t))
    if norm == 'l2':
        loss_obs = loss_obs * loss_obs
        loss_cen = loss_cen * loss_cen
    loss_recon = loss_obs + loss_cen
    _alpha = alpha if cur_alpha is None else cur_alpha
    loss = (1.0 - _alpha) * loss_recon + _alpha * loss_obs
    loss = loss.mean()
    return loss

class RankLoss(nn.Module):
    def __init__(self):
        super(RankLoss, self).__init__()
        self.gamma = 1.0
        self.norm = 'l1'
        self.add_weight = False

    def forward(self, pred_t, t, e):
        return rank_loss(pred_t, t, e, self.gamma, self.norm, self.add_weight)

def rank_loss(pred_t, t, e, gamma=1, norm='l1', add_weight=False):
    """Continuous Survival Model

    Ranking loss for preditions and observations.
    for pairs (i, j) conditioned on e_i = 1 & t_i < t_j:
        diff_ij = (-pred_t_i) - (-pred_t_j)
        rank_loss = ||max(0, gamma - diff_ij)||_norm
                  = ||max(0, gamma + pred_t_i - pred_t_j)||_norm
    """
    # pred_t = pred_t.squeeze()
    hazards = torch.sigmoid(pred_t)
    survival = torch.cumprod(1 - hazards, dim=1)
    pred_t = -torch.sum(survival, dim=1)
    t = t.squeeze()
    e = e.squeeze()
    pair_mask = (t.view(-1, 1) < t.view(1, -1)) * (e.view(-1, 1) == 1)
    if not torch.any(pair_mask):
        return torch.Tensor([0.0]).to(pred_t.device)
    pair_diff = pred_t.view(-1, 1) - pred_t.view(1, -1) # the lower, the best
    pair_loss = F.relu(gamma + pair_diff)
    pair_mask = pair_mask.float()
    if add_weight:
        # masked_log_softmax
        x = pair_diff
        maxx = (x * pair_mask + (1 - 1 / (pair_mask + 1e-5))).max()
        log_ex = x - maxx
        log_softmax = log_ex - (torch.exp(log_ex * pair_mask) * pair_mask).sum().log()
        normed_weight = (log_softmax * pair_mask).exp() * pair_mask
    else:
        weight = pair_mask
        normed_weight = weight / weight.sum()

    if norm == 'l2':
        pair_loss = pair_loss * pair_loss
    elif norm == 'l1':
        pass
    else:
        raise NotImplementedError('Arg. `norm` expected l1/l2, but got {}'.format(norm))

    rank_loss = (pair_loss * normed_weight).sum()
    return rank_loss

def MSE_loss(pred_t, t, e, include_censored=False):
    """Continuous Survival Model.

    MSE loss for pred_t and labels, used for reproducing ESAT (shen et al., ESAT, AAAI, 2022).
    Please refer to its official repo: https://github.com/notbadforme/ESAT/blob/main/esat/trainforesat.py#L111
    """
    pred_t = pred_t.squeeze()
    t = t.squeeze()
    e = e.squeeze()
    loss = e * (pred_t - t) * (pred_t - t)
    if include_censored:
        loss += (1 - e) * (pred_t - t) * (pred_t - t)
    loss = loss.mean()
    return loss

class SurvMLE(nn.Module):
    """A maximum likelihood estimation function in Survival Analysis.
    As suggested in '10.1109/TPAMI.2020.2979450',
        [*] L = (1 - alpha) * loss_l + alpha * loss_z.
    where loss_l is the negative log-likelihood loss, loss_z is an upweighted term for instances
    D_uncensored. In discrete model, T = 0 if t in [0, a_1), T = 1 if t in [a_1, a_2) ...
    The larger the alpha, the bigger the importance of event_loss.
    If alpha = 0, event loss and censored loss are viewed equally.
    This implementation is based on https://github.com/mahmoodlab/MCAT/blob/master/utils/utils.py
    """
    def __init__(self, alpha=0.0, eps=1e-7):
        super(SurvMLE, self).__init__()
        self.alpha = alpha
        self.eps = eps
        print('[setup] loss: a MLE loss in discrete SA models with alpha = %.2f' % self.alpha)

    def forward(self, hazards_hat, t, e, cur_alpha=None):
        """
        y: torch.FloatTensor() with shape of [B, 2] for a discrete model.
        t: torch.LongTensor() with shape of [B, ] or [B, 1]. It's a discrete time label.
        e: torch.FloatTensor() with shape of [B, ] or [B, 1].
            e = 1 for uncensored samples (with event),
            e = 0 for censored samples (without event).
        hazards_hat: torch.FloatTensor() with shape of [B, MAX_T]
        """
        batch_size = len(t)
        t = t.view(batch_size, 1).long() # ground truth bin, 0 [0,a_1), 1 [a_1,a_2),...,k-1 [a_k-1,inf)
        c = 1 - e.view(batch_size, 1).float() # convert it to censorship status, 0 or 1
        S = torch.cumprod(1 - hazards_hat, dim=1) # surival is cumulative product of 1 - hazards
        S_padded = torch.cat([torch.ones_like(c), S], 1) # s[0] = 1.0 to avoid for t = 0
        uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, t).clamp(min=self.eps)) + torch.log(torch.gather(hazards_hat, 1, t).clamp(min=self.eps)))
        censored_loss = - c * torch.log(torch.gather(S_padded, 1, t+1).clamp(min=self.eps))
        neg_l = censored_loss + uncensored_loss
        alpha = self.alpha if cur_alpha is None else cur_alpha
        loss = (1.0 - alpha) * neg_l + alpha * uncensored_loss
        loss = loss.mean()
        return loss


class SurvPLE(nn.Module):
    """A partial likelihood estimation (called Breslow estimation) function in Survival Analysis.

    This is a pytorch implementation by Huang. See more in https://github.com/huangzhii/SALMON.
    Note that it only suppurts survival data with no ties (i.e., event occurrence at same time).

    Args:
        y_hat (Tensor): Predictions given by the survival prediction model.
        T (Tensor): The last observed time.
        E (Tensor): An indicator of event observation.
            if E = 1, uncensored one (with event)
            else E = 0, censored one (without event)
    """

    def __init__(self):
        super(SurvPLE, self).__init__()
        self.CONSTANT = torch.tensor(10.0)
        print('[setup] loss: a popular PLE loss in coxph')

    def forward(self, y_hat, T, E):
        y_hat = torch.sigmoid(y_hat)
        device = y_hat.device
        # numerical overflow
        cont = self.CONSTANT.to(device)
        y_hat = torch.where(y_hat > cont, cont, y_hat)

        n_batch = len(T)
        R_matrix_train = torch.zeros([n_batch, n_batch], dtype=torch.int8)
        for i in range(n_batch):
            for j in range(n_batch):
                R_matrix_train[i, j] = T[j] >= T[i]

        train_R = R_matrix_train.float().to(device)
        train_ystatus = E.float().to(device)

        theta = y_hat.reshape(-1)
        exp_theta = torch.exp(theta)

        loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

        return loss_nn

##############################################
# General Loss for Discriminator and Generator
##############################################
def real_fake_loss(real, fake, which='bce'):
    fake = fake.squeeze()
    if which == 'bce':
        fake = torch.sigmoid(fake)
        loss = - torch.mean(1.0 - torch.log(fake + 1e-8))
        if real is not None:
            real = real.squeeze()
            real = torch.sigmoid(real)
            loss = loss - torch.mean(torch.log(real + 1e-8))
    elif which == 'hinge':
        loss = nn.ReLU()(1.0 + fake).mean()
        if real is not None:
            real = real.squeeze()
            loss = loss + nn.ReLU()(1.0 - real).mean()
    elif which == 'wasserstein':
        loss = fake.mean()
        if real is not None:
            real = real.squeeze()
            loss = loss - real.mean()
    else:
        loss = None
    return loss


def fake_generator_loss(fake_score):
    # using the value before applying sigmoid -> fake = sigmoid(fake_score)
    fake_score = fake_score.squeeze()
    return - torch.mean(fake_score)