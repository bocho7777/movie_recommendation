#3) TMDM_Loss
 #3.1) using BCE_Loss

def TMDM_Loss(logit, target):
  loss_function = nn.BCELoss()
  loss = loss_function(logit, target)

  return loss
