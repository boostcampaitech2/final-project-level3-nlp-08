import torch
from torch.utils.data import (DataLoader, RandomSampler)
from torch.nn import MSELoss

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer

from tqdm import trange, tqdm

import pickle

from scipy.stats import pearsonr

def train(args, train_dataset, val_u_seqs, val_v_seqs, validation_labels, sen_encoder):

    # logging
    best_cor = 0
    stop_counter = 0 

    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_bs)

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
          {'params': [p for n, p in sen_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
          {'params': [p for n, p in sen_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
          ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Start training!
    global_step = 0
    
    sen_encoder.zero_grad()
    torch.cuda.empty_cache()
    
    train_iterator = trange(int(args.num_epochs), desc="Epoch")

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        # to compute average loss in an epoch
        train_loss_list = []

        print(f"**********Train: epoch {epoch}**********")
        for step, batch in enumerate(epoch_iterator):
            
            sen_encoder.train()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            inputs = {'input_ids_1': batch[0],
                        'input_ids_2': batch[1],
                        'attention_mask_1': batch[2],
                        'attention_mask_2': batch[3]
                       }

            targets = batch[4]
            cos_sim_outputs = (sen_encoder(**inputs) * (5/2)) + 2.5

            # loss
            criterion = MSELoss()
            train_loss = criterion(cos_sim_outputs, targets.float())
            train_loss_list.append(train_loss.detach().cpu().numpy())

            # print loss every 1000 steps
            if step % 500 == 0 and step > 99:
                epoch_average_loss = sum(train_loss_list[step-100:step]) / 99
                print(f'step: {step} with loss: {epoch_average_loss}')

            train_loss = train_loss / args.gradient_accumulation_steps
            train_loss.backward()

            if ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == len(epoch_iterator)):
                optimizer.step()
                scheduler.step()
                sen_encoder.zero_grad()

            

            global_step += 1
            
            torch.cuda.empty_cache()
        
        print("**********EVALUATION**********")
        with torch.no_grad():
            sen_encoder.eval()
            sen_encoder.eval()

            u_input_ids_lst = val_u_seqs['input_ids']
            u_attention_mask_lst = val_u_seqs['attention_mask']

            v_input_ids_lst =  val_v_seqs['input_ids']
            v_attention_mask_lst = val_v_seqs['attention_mask']

            val_cos_sim_lst = []

            for i in range(0,len(validation_labels)):
                u_input_ids = torch.tensor(u_input_ids_lst[i]).cuda().unsqueeze(0)
                u_attention_mask = torch.tensor(u_attention_mask_lst[i]).cuda().unsqueeze(0)
                v_input_ids = torch.tensor(v_input_ids_lst[i]).cuda().unsqueeze(0)
                v_attention_mask = torch.tensor(v_attention_mask_lst[i]).cuda().unsqueeze(0)

                inputs = {'input_ids_1': u_input_ids,
                        'input_ids_2': v_input_ids,
                        'attention_mask_1': u_attention_mask,
                        'attention_mask_2': v_attention_mask
                       }

                val_cos_sim = sen_encoder(**inputs).to('cpu')
                val_cos_sim_lst.append(val_cos_sim.item()*5/2+2.5)   #(num_query, emb_dim)

        pearson_cor = pearsonr(validation_labels.tolist(), val_cos_sim_lst)[0]

        if pearson_cor > best_cor:
            stop_counter = 0
            best_cor = pearson_cor
        
            sen_encoder.save_pretrained(f'checkpoints/val_acc_{pearson_cor:4.2%}_sen_encoder')
                
        else:
            stop_counter += 1
            print(f"early stop count {stop_counter} out of {args.early_stop}")
            if args.early_stop == stop_counter:
                break

        print("epoch pearson correaltion:", pearson_cor)
        print("best cor from all epochs", best_cor)
