import numpy as np
import torch
from tqdm._tqdm import tqdm
import os
import random
import gc
import argparse
from model_gpt2 import Model
import copy
import time

from transformers import AdamW
from transformers import GPT2Tokenizer, GPT2LMHeadModel


torch.cuda.set_device(0)
torch.backends.cudnn.deterministic = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


def load_data(files, max_len=50, eos='<eos>'):
    dataset = {'train': {}, 'dev': {}}
    for set_info, f in files.items():
        with open(f) as fin:
            input_data = []
            target = []
            generation_seeds = []
            for line in fin:
                row = line.strip().split("\t")
                input_data.append(row[1].split()
                                  + ['<sep>']
                                  + row[0].split() + [eos])
                generation_seeds.append(row[1].split() + ['<sep>'])
                target.append(row[0].split() + [eos])

            dataset[set_info]['input'] = input_data
            dataset[set_info]['target'] = target
            dataset[set_info]['seed'] = generation_seeds

    return dataset


def prepare(dataset, tokenizer):
    data_input = {}
    gen_seed = {}
    target = {}
    target_weights = {}
    for set_info, data in dataset.items():
        data_input[set_info] = []
        gen_seed[set_info] = []
        target[set_info] = []
        target_weights[set_info] = []
        for input_text in data['input']:
            data_input[set_info].append(tokenizer.encode(" ".join(input_text)))

            weights = []
            tag_in = False
            for i, t in enumerate(tokenizer.tokenize(" ".join(input_text))):
                if t == '<pre>' or t == '<event>':
                    tag_in = True
                if tag_in:
                    weights.append(1.)
                else:
                    weights.append(1.)
                if t == '</pre>' or t == '</event>':
                    tag_in = False
            target_weights[set_info].append(weights)

        for input_text in data['seed']:
            gen_seed[set_info].append(tokenizer.encode(" ".join(input_text)))
        for input_text in data['target']:
            target[set_info].append(tokenizer.encode(" ".join(input_text)))

    return data_input, gen_seed, target, target_weights


def batchify(enc_input, batch_size=16):
    batch_data = []
    buf_e = []
    for e in enc_input:
        buf_e.append(e)
        if len(buf_e) == batch_size:
            batch_data.append(buf_e)
            buf_e = []

    if buf_e:
        batch_data.append(buf_e)

    return batch_data


def main(args):

    print("Load Data")
    print(args.train_data, args.dev_data)
    files = {'train': args.train_data, 'dev': args.dev_data}

    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token='<PAD>')

    # add tokens for precondition generation
    tokenizer.add_tokens(
            ['<sep>', '<event>', '</event>',
                '<pre>', '</pre>', '<eos>', '[BLANK]'])
    encdec = GPT2LMHeadModel.from_pretrained(model_name)
    encdec.resize_token_embeddings(len(tokenizer))

    # dataset load
    dataset = load_data(files, max_len=args.max_sequence_length, eos='<eos>')

    if args.load_model is not None:
        model = torch.load(args.load_model)
    else:
        model = Model(tokenizer, encdec)
        if model.use_cuda:
            model.cuda()

    data_input, gen_seed, target, target_weights = prepare(dataset, tokenizer)

    # Set a path for saving model
    save_model_path = os.path.join(
            args.save_model_path,
            args.experiment)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, eps=1e-8)

    n_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("#parameters: {}".format(n_params))

    N = len(data_input['train'])
    print(N//args.batch_size)
    best_dev_loss = 9999
    for epoch in range(1, args.epochs+1):
        print("Epoch {}:".format(epoch))
        start_time = time.time()
        batch_idxs = np.random.permutation(N//args.batch_size+1)
        line_tqdm = tqdm(batch_idxs, dynamic_ncols=True)
        total_loss = []
        model.train()

        for batch_idx in line_tqdm:
            enc_input = data_input['train'][batch_idx*args.batch_size:min((batch_idx+1)*args.batch_size, N)]
            tmp = gen_seed['train'][batch_idx*args.batch_size:min((batch_idx+1)*args.batch_size, N)]
            event_lens = [len(s) for s in tmp]

            if len(enc_input) == 0:
                continue

            model.zero_grad()

            loss = model(
                    enc_input,
                    copy.deepcopy(enc_input),
                    event_lens)

            total_loss.append(loss.data.cpu().numpy().tolist())
            loss.backward()
            optimizer.step()
            gc.collect()
            torch.cuda.empty_cache()

        end_time = time.time()
        print("Time elapsed: {:.3f}".format(end_time-start_time))
        print("Loss: {}".format(sum(total_loss)/len(total_loss)))

        model.eval()
        with torch.no_grad():
            for set_info in ['train', 'dev']:
                NN = len(data_input[set_info])
                total_loss = []
                for idx in range(NN//args.batch_size):
                    enc_input = data_input[set_info][idx*args.batch_size:min((idx+1)*args.batch_size, NN)]
                    tmp = gen_seed[set_info][idx*args.batch_size:min((idx+1)*args.batch_size, NN)]
                    event_lens = [len(s) for s in tmp]

                    if len(enc_input) == 0:
                        continue

                    loss = model(
                            enc_input,
                            copy.deepcopy(enc_input),
                            event_lens)

                    total_loss.append(loss.data.cpu().numpy().tolist())

                loss = sum(total_loss) / len(total_loss)
                print("Test on {} set:".format(set_info))
                print("\tLoss: {}".format(loss))
                if set_info == 'dev':
                    if best_dev_loss > loss:
                        best_dev_loss = loss
                        torch.save(
                                model,
                                os.path.join(
                                    save_model_path, "DevBest.pt"))

            for d, t in zip(gen_seed['dev'][:10], target['dev'][:10]):
                sent = model.generate(d)
                print("Target Event: ", tokenizer.decode(d))
                print("Generated Precondition: ", sent)
                print("Reference: ", tokenizer.decode(t))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data', type=str, default='../data/peko_gen_train.txt')
    parser.add_argument('--dev_data', type=str, default='../data/peko_gen_dev.txt')
    parser.add_argument('--max_sequence_length', type=int, default=54)

    parser.add_argument('-ep', '--epochs', type=int, default=100)
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.000001)

    parser.add_argument('-bin','--save_model_path', type=str, default='/home/data/PrecondGen/')

    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('-ex', '--experiment', type=str, default='test')

    args = parser.parse_args()

    main(args)

