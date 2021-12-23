import torch
from torch.utils.data import (DataLoader, RandomSampler)
from torch.nn import MSELoss, CrossEntropyLoss

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer

from tqdm import trange, tqdm

from scipy.stats import pearsonr

import numpy as np

def train(args, train_dataset, val_emb, validation_labels, scoring_model):

    # logging
    best_cor = 0
    stop_counter = 0 

    # Dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_bs, drop_last=True)

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
          {'params': [p for n, p in scoring_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
          {'params': [p for n, p in scoring_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
          ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Start training!
    global_step = 0
    
    scoring_model.zero_grad()
    torch.cuda.empty_cache()
    
    train_iterator = trange(int(args.num_epochs), desc="Epoch")

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        # to compute average loss in an epoch
        train_loss_list = []

        print(f"**********Train: epoch {epoch}**********")
        for step, batch in enumerate(epoch_iterator):
            
            scoring_model.train()

            if torch.cuda.is_available():
                batch = tuple(t.cuda() for t in batch)

            inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1]
                       }

            scores = scoring_model(**inputs, labels=batch[2])

            train_loss = scores.loss

            # loss
            #criterion = MSELoss()
            #train_loss = criterion(answer_label.float(), scores.float())
            train_loss_list.append(train_loss.detach().cpu().numpy())

            # print loss every 1000 steps
            if step % 100 == 0 and step > 99:
                epoch_average_loss = sum(train_loss_list[step-100:step]) / 99
                print(f'step: {step} with loss: {epoch_average_loss}')

            train_loss = train_loss / args.gradient_accumulation_steps
            train_loss.backward()

            if ((step + 1) % args.gradient_accumulation_steps == 0) or (step + 1 == len(epoch_iterator)):
                optimizer.step()
                scheduler.step()
                scoring_model.zero_grad()

            global_step += 1
            
            torch.cuda.empty_cache()
        
        print("**********EVALUATION**********")
        with torch.no_grad():
            scoring_model.eval()

            input_ids_lst = val_emb['input_ids']
            attention_mask_lst = val_emb['attention_mask']

            val_result_lst = []
            val_loss_lst = []

            for i in range(0,len(validation_labels)):
                input_ids = torch.tensor(input_ids_lst[i]).cuda().unsqueeze(0)
                attention_mask = torch.tensor(attention_mask_lst[i]).cuda().unsqueeze(0)
                label = torch.tensor(validation_labels[i]).cuda().unsqueeze(0)

                val_output = scoring_model(input_ids, attention_mask, labels=label)
                val_loss_lst.append(val_output.loss.cpu())
                val_result = np.argmax(val_output.logits.cpu())
                val_result_lst.append(val_result)   #(num_query, emb_dim)

        print(validation_labels, val_result_lst)
        pearson_cor = pearsonr(validation_labels, val_result_lst)[0]
        val_loss = (sum(val_result_lst) / len(validation_labels)).item()

        if pearson_cor > best_cor:
            stop_counter = 0
            best_cor = pearson_cor
        
            scoring_model.save_pretrained(f'checkpoints/val_cor_{pearson_cor:4.2%}_sen_encoder')
                
        else:
            stop_counter += 1
            print(f"early stop count {stop_counter} out of {args.early_stop}")
            if args.early_stop == stop_counter:
                break

        print("epoch loss:", val_loss)
        print("epoch pearson correaltion:", pearson_cor)
        print("best cor from all epochs", best_cor)
