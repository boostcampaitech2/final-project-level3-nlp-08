from transformers import RobertaModel, RobertaPreTrainedModel
from torch import mean, nn
import torch

class RobertaScorer(nn.Module):
    def __init__(self, model_name):
      super(RobertaScorer, self).__init__()
      
      self.roberta = RobertaModel.from_pretrained(model_name)
      self.lstm = nn.LSTM(1024, 1024, batch_first=True,bidirectional=True).cuda()
      self.linear1 = nn.Linear(1024*2, 512)
      self.bn1 = nn.BatchNorm1d(512)
      self.dropout1 = nn.Dropout(p=0.5)
    
      self.linear2 = nn.Linear(512, 128)
      self.bn2 = nn.BatchNorm1d(128)
      self.dropout2 = nn.Dropout(p=0.5)
      
      self.linear3 = nn.Linear(128, 32)
      self.bn3 = nn.BatchNorm1d(32)
      self.dropout3 = nn.Dropout(p=0.5)

      self.linear4 = nn.Linear(32, 1)
      self.sig = nn.Sigmoid()

      
    def forward(self, input_ids, attention_mask=None): 
  
        outputs = self.roberta(input_ids, attention_mask)

        seq_output = outputs[0]

        lstm_output, (h,c) = self.lstm(seq_output)
        hidden = torch.cat((lstm_output[:,-1, :1024],lstm_output[:,0, 1024:]),dim=-1)
        x = self.linear1(hidden.view(-1,1024*2))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.linear4(x)

        return self.sig(x)
