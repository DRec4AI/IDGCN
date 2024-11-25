import torch.nn as nn
import torch
import torch.nn.functional as F

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """
        pos_logits = y_pred[:, 0]
        pos_loss = torch.pow(pos_logits - 1, 2) / 2
        neg_logits = y_pred[:, 1:]
        neg_loss = torch.pow(neg_logits, 2).sum(dim=-1) / 2
        loss = pos_loss + neg_loss
        return loss.mean()


class PairwiseLogisticLoss(nn.Module):
    def __init__(self):
        super(PairwiseLogisticLoss, self).__init__()

    def forward(self, pos_logits, neg_logits):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        if pos_logits.dim()==1:
            pos_logits = pos_logits.unsqueeze(-1)
        if neg_logits.dim()==1:
            neg_logits = neg_logits.unsqueeze(-1)

        logits_diff = pos_logits - neg_logits

        loss = -torch.log(torch.sigmoid(logits_diff))
        loss = loss.mean()
        return loss
class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """

        return torch.mean(F.softplus(neg_score - pos_score))
        return loss

class GumbelLoss(nn.Module):

    def __init__(self):
        super(GumbelLoss, self).__init__()

    def forward(self, pos_logits, neg_logits):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        if pos_logits.dim()==1:
            pos_logits = pos_logits.unsqueeze(-1)
        if neg_logits.dim()==1:
            neg_logits = neg_logits.unsqueeze(-1)

        logits_diff = pos_logits - neg_logits

        loss = torch.exp(-logits_diff)

        reduction = True
        if reduction:
            loss = loss.mean()
        else:
            loss = (loss.sum(dim=-1)).sum(dim=-1)
        return loss

class MarginalHingeLoss(nn.Module):
    def __init__(self, gama=0):
        super(MarginalHingeLoss, self).__init__()
        self.gama = gama

    def forward(self, pos_logits, neg_logits):
        """
        :param y_true: Labels
        :param y_pred: Predicted result.
        """
        pos_logits = pos_logits.unsqueeze(-1)
        logits_diff = -pos_logits + neg_logits + self.gama
        values,_ = torch.max(logits_diff, 1)
        loss = values.mean()
        return loss

class InfoNCELoss(nn.Module):
    def __init__(self,temp=1.0):
        """
        :param num_negs: number of negative instances in bpr loss.
        """
        super(InfoNCELoss, self).__init__()
        self.temp = temp
        print("temperture:", temp)


    def forward(self, pos_logits, neg_logits, weight=None):
        if pos_logits.dim()==1:
            pos_logits = pos_logits.unsqueeze(-1)
        logits = torch.cat((pos_logits,neg_logits),dim=1)
        hit_probs = F.softmax(logits/self.temp, dim=1)
        hit_probs = hit_probs[:,0]

        if weight!=None:
            hit_probs = weight.squeeze(1)*hit_probs
        loss = -torch.log(hit_probs).mean()
        return loss

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        print("temperature: ", self.temperature)

    def forward(self, anchor, items):
        # anchor: [B, dim]     user
        # items:  [B, dim]
        anchor, items = F.normalize(anchor, dim=-1), F.normalize(items, dim=-1)  #normalization
        pos_score = (anchor * items).sum(dim=-1)  #(B,) sum is a number
        pos_score = torch.exp((pos_score) / self.temperature)    #e^x+
        ttl_score = torch.matmul(anchor, items.transpose(0, 1))  # (b,emb)  (emb,b) = (b, b)
        ttl_score_user = torch.exp((ttl_score) / self.temperature).sum(dim=1)   #negative contrastive (b, )   e^x-
        cl_loss_user = - torch.log(pos_score / ttl_score_user)
        return torch.mean(cl_loss_user)

class UniformLoss(nn.Module):
    def __init__(self, uniform_weight):
        super(UniformLoss, self).__init__()
        self.temperature = 0.1
        print("temperature: ", self.temperature)
        self.uniform_weight = uniform_weight
        print("uniform_weight:", uniform_weight)


    def forward(self, anchor, items, temp=None):
        if temp is None:
            temp = self.temperature
        # anchor: [B, dim]
        # items:  [B, dim]

        anchor, items = F.normalize(anchor, dim=-1), F.normalize(items, dim=-1)
        pos_score = (anchor * items).sum(dim=-1)


        ttl_score = torch.matmul(anchor, items.transpose(0, 1))

        pos_score = torch.exp((pos_score) / temp)
        ttl_score_user = torch.exp((ttl_score) / temp).sum(dim=1)
        u_i_loss = -torch.log(pos_score / ttl_score_user).mean()

        return u_i_loss

class DirectAU(nn.Module):
    def __init__(self,gamma):
        super(DirectAU, self).__init__()
        self.gamma = gamma
        print("gamma:",self.gamma)



    def forward(self, anchor, items):
        # anchor: [B, dim]
        # items:  [B, dim]
        anchor, items = F.normalize(anchor, dim=-1), F.normalize(items, dim=-1)


        align_loss = 0

        anchor_uni = (torch.matmul(anchor, anchor.transpose(0, 1))/0.1).exp().mean().log()
        items_uni = (torch.matmul(items, items.transpose(0, 1)) / 0.1).exp().mean().log()

        uni_loss =  anchor_uni + items_uni

        loss = align_loss + uni_loss*self.gamma

        return loss


class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0, negative_weight=None):
        """
        :param margin: float, margin in CosineContrastiveLoss
        :param num_negs: int, number of negative samples
        :param negative_weight:, float, the weight set to the negative samples. When negative_weight=None, it
            equals to num_negs
        """
        super(CosineContrastiveLoss, self).__init__()
        self._margin = margin
        self._negative_weight = negative_weight
        print(margin,negative_weight)

    def forward(self, y_pred, y_true):
        """
        :param y_pred: prdicted values of shape (batch_size, 1 + num_negs)
        :param y_true: true labels of shape (batch_size, 1 + num_negs)
        """

        pos_logits = y_pred[:, 0]
        pos_loss = torch.relu(1 - pos_logits)
        neg_logits = y_pred[:, 1:]
        neg_loss = torch.relu(neg_logits - self._margin)
        if self._negative_weight:
            loss = pos_loss + neg_loss.mean(dim=-1) * self._negative_weight
        else:
            loss = pos_loss + neg_loss.sum(dim=-1)
        return loss.mean()