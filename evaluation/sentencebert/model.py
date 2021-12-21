from transformers import RobertaModel, RobertaPreTrainedModel
from torch import mean, nn

class RobertaEncoder(RobertaPreTrainedModel):
    def __init__(self, config):
      super(RobertaEncoder, self).__init__(config)
      
      self.roberta = RobertaModel(config)
      self.cos_sim = nn.CosineSimilarity(dim=1)
      self.init_weights()
      
    def forward(self, input_ids_1, input_ids_2, attention_mask_1=None, attention_mask_2=None): 
  
        outputs_1 = self.roberta(input_ids_1 ,attention_mask=attention_mask_1)
        outputs_2 = self.roberta(input_ids_2 ,attention_mask=attention_mask_2)

        sequence_outputs_1 = outputs_1[0]
        sequence_outputs_2 = outputs_2[0]
        
        u = mean(sequence_outputs_1,1)
        v = mean(sequence_outputs_2,1)
        
        cos_sim = self.cos_sim(u,v)

        return cos_sim
