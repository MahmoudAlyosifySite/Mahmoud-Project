"""Contrastive losses.

One function covers SimCLR, SupCon and the semi-supervised union of the two.
Every view carries a *group id*; two different views are positives if and only
if they share a group id:

* unlabeled image  -> the group is the image itself (id = C + image index);
* labeled image    -> the group is its class        (id = class label).

With no labels this is NT-Xent (SimCLR). With all labels it is the SupCon loss
(L_out of Khosla et al. 2020). With some labels it is SupCon over the whole
pool in which every unlabeled image is its own class.
"""

import torch
import torch.nn.functional as F


def make_groups(index, labels, labeled_mask, num_classes):
    """Group ids for a batch.

    index        [B] image indices in the training set
    labels       [B] class labels (only read where ``labeled_mask`` is True)
    labeled_mask [B] True where the label may be used
    """
    return torch.where(labeled_mask, labels, num_classes + index)


def grouped_contrastive_loss(z, groups, temperature):
    """Supervised-contrastive loss with positives defined by ``groups``.

    z           [V*B, D] L2-normalised embeddings, the V views concatenated view by view
    groups      [B] group id per image
    temperature float, or a tensor [B] with one temperature per anchor image
    """
    z = z.float()
    n_views = z.shape[0] // groups.shape[0]
    g = groups.repeat(n_views)
    if torch.is_tensor(temperature):
        temperature = temperature.repeat(n_views).view(-1, 1)
    logits = z @ z.t() / temperature
    self_mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos = (g.view(-1, 1) == g.view(1, -1)) & ~self_mask
    # every anchor has at least one positive: its other view
    loss = -(log_prob.masked_fill(~pos, 0.0)).sum(1) / pos.sum(1)
    return loss.mean()


def negative_cosine(p, z):
    """SimSiam / BYOL regression loss between a prediction and a stop-gradient target."""
    return -(F.normalize(p.float(), dim=1) * F.normalize(z.float().detach(), dim=1)).sum(1).mean()
