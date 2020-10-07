import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from tqdm._tqdm import tqdm
import os
import random
import json
import gc
import argparse
import time

from transformers import AdamW
from transformers import BertModel, BertTokenizer


torch.cuda.set_device(0)
torch.backends.cudnn.deterministic = True


class Model(nn.Module):
    def __init__(self, tokenizer, encoder, embedding_dim, n_class):
        super(Model, self).__init__()
        self.use_cuda = True if torch.cuda.is_available() else False
        self.embedding_dim = embedding_dim

        self.tokenizer = tokenizer
        self.encoder = encoder

        self.output = nn.Linear(self.embedding_dim*2, n_class)
        self.softmax = nn.LogSoftmax(dim=-1)

    def get_var(self, tensor):
        if self.use_cuda:
            return Variable(tensor.cuda())
        else:
            return Variable(tensor)

    def encode(self, indexed_tokens):
        """
        Encode the sequence input
        The input is a list of indexed tokens.
        This function will automatically pad sequences and convert them into tensor (batch_size) x (seq_length).
        Also, token_type_ids and attention mask are derived automatically.
        The output will be contextualized embeddings (batch_size) x (seq_lenght) x (embedding_dim).
        """
        max_len = max([len(ids) for ids in indexed_tokens]) + 2
        tokens_tensor = []
        token_type_ids = []
        attention_mask = []
        for instance in indexed_tokens:
            encoded_input = self.tokenizer.prepare_for_model(
                                                instance,
                                                max_length=max_len,
                                                pad_to_max_length=True)
            tokens_tensor.append(encoded_input['input_ids'])
            token_type_ids.append(encoded_input['token_type_ids'])
            attention_mask.append(encoded_input['attention_mask'])

        tokens_tensor = torch.tensor(tokens_tensor)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)

        if self.use_cuda:
            tokens_tensor = tokens_tensor.cuda()
            token_type_ids = token_type_ids.cuda()
            attention_mask = attention_mask.cuda()

        return self.encoder(
                    input_ids=tokens_tensor,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask)[0]

    def forward(self, sentences, relation):
        """
        This model will return a log-probability (output from logsoftmax) of precondition relation.
        The input is a list of sentences -- a sentence is represented as a list of token indcies -- and a pair of event indices.
        The output will be (batch_size) x (n_class=2), 0: NOT being a precondition, 1: being a precondition.
        """

        sent_output = self.encode(sentences)

        batch_size, seq_len, dim = sent_output.size()

        rel_repr = []
        for para, rels in zip(sent_output, relation):
            e1, e2 = rels
            e1_idx = torch.arange(e1[0], e1[1])
            e2_idx = torch.arange(e2[0], e2[1])
            e1_repr = torch.sum(para.index_select(
                                    0, self.get_var(e1_idx)), dim=0)
            e2_repr = torch.sum(para.index_select(
                                    0, self.get_var(e2_idx)), dim=0)

            rel_repr.append(torch.cat((e1_repr, e2_repr)))

        rel_repr = torch.stack(rel_repr, dim=0)
        e1_repr, e2_repr = torch.chunk(rel_repr, 2, dim=1)

        logits = self.output(rel_repr)

        return self.softmax(logits)


def load_data(filename):

    def split(data):
        L = len(data)
        idx = np.random.permutation(L)
        train = [data[i] for i in idx[:L//10*6]]
        dev = [data[i] for i in idx[L//10*6:L//10*8]]
        test = [data[i] for i in idx[L//10*8:]]

        return train, dev, test

    pos_data = []
    neg_data = []
    with open(filename, "r") as fin:
        for line in fin:
            line_in = json.loads(line.strip())
            if line_in['label'] == 1:
                pos_data.append(line_in)
            else:
                neg_data.append(line_in)

    pos_train, pos_dev, pos_test = split(pos_data)
    neg_train, neg_dev, neg_test = split(neg_data)
    train = pos_train+neg_train
    dev = pos_dev+neg_dev
    test = pos_test+neg_test

    train = [train[i] for i in np.random.permutation(len(train))]
    dev = [dev[i] for i in np.random.permutation(len(dev))]
    test = [test[i] for i in np.random.permutation(len(test))]

    return {'train': train, 'dev': dev, 'test': test}


def prepare(data, tokenizer):
    paragraphs = []
    relations = []
    labels = []
    for row in data:

        sent = row['sent'].split()

        tokens = tokenizer.tokenize(" ".join(sent))
        i, j, start_idx = 0, 0, 0
        new_idxs = []
        text_buf = []
        while i < len(sent):
            if sent[i] == " "*len(sent[i]):
                i += 1
                new_idxs.append(0)
            else:
                break
        while i < len(sent) and j < len(tokens):
            text_buf.append(tokens[j])
            if tokenizer.convert_tokens_to_string(text_buf) == tokenizer.convert_tokens_to_string(tokenizer.tokenize(sent[i])):
                i += 1
                new_idxs.append(start_idx)
                start_idx = j+1
                text_buf = []
            j += 1
        new_idxs.append(len(tokens))

        paragraphs.append(tokenizer.convert_tokens_to_ids(tokens))

        relations.append([[new_idxs[ii]+1 for ii in row['source']['idx']], [new_idxs[ii]+1 for ii in row['target']['idx']]])
        labels.append(row['label'])

    return paragraphs, relations, labels


def getScores(pred, label):

    true_positive = sum([1 if p * l else 0 for p, l in zip(pred, label)])
    false_positive = sum(pred) - true_positive
    false_negative = sum(label) - true_positive
    true_negative = len(pred) - (true_positive + false_positive + false_negative)

    precision = true_positive / (true_positive+false_positive+1e-10)
    recall = true_positive / (true_positive+false_negative+1e-10)

    f1 = 2*precision*recall / (precision + recall + 1e-10)

    acc = (true_positive+true_negative)/len(pred)

    return acc, precision, recall, f1


def test(model, test_para, test_relation, test_label, loss_function=None, pred_flag=False):
    total_loss = []
    model.eval()
    pred = []

    N = len(test_para)
    test_size = 32
    line_tqdm = tqdm(range(N//test_size + 1), dynamic_ncols=True)
    for i in line_tqdm:
        para_test = test_para[i*test_size:min((i+1)*test_size, N)]
        relation_test = test_relation[i*test_size:min((i+1)*test_size, N)]
        label_test = test_label[i*test_size:min((i+1)*test_size, N)]
        score = model(para_test, relation_test)
        target = Variable(torch.LongTensor(label_test))
        if model.use_cuda:
            target = target.cuda()

        if loss_function is not None:
            loss = loss_function(score, target)
            total_loss.extend(loss.data.cpu().numpy().tolist())

        pred.extend(torch.argmax(score, dim=-1).cpu().tolist())

    acc, precision, recall, f1 = getScores(pred, test_label)

    if loss_function is not None:
        print("\t\tLoss: {:0.5f}".format(sum(total_loss)/len(total_loss)))
    print("\t\tAccuracy: {:0.5f}".format(acc))
    print("\t\tPrecision: {:0.5f}".format(precision))
    print("\t\tRecall: {:0.5f}".format(recall))
    print("\t\tF1: {:0.5f}".format(f1))

    if loss_function is not None:
        out = (acc, precision, recall, f1, sum(total_loss)/len(total_loss))
    else:
        out = (acc, precision, recall, f1)

    if pred_flag:
        out = (pred,) + out

    return out


def batchify(paragraphs, relations, labels, batch_size=16):

    batch_data = []
    para_batch = []
    rel_batch = []
    l_batch = []
    for para, relation, label in zip(paragraphs, relations, labels):
        para_batch.append(para)
        rel_batch.append(relation)
        l_batch.append(label)

        if len(l_batch) == batch_size:
            batch_data.append((para_batch, rel_batch, l_batch))
            para_batch = []
            rel_batch = []
            l_batch = []

    if len(l_batch) != 0:
        batch_data.append((para_batch, rel_batch, l_batch))

    return batch_data


def weight_check(m):
    if type(m) == nn.Linear:
        print(m.weight)


def train(args):

    out_dir = os.path.join(args.logdir, args.experiment)

    # setup tensorboard logging
    if args.tensorboard_logging:
        writer = SummaryWriter(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load data
    data = load_data('../data/peko_all.jsonl')
    del data['test']

    # load transformer tokenizer, model
    model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name, pad_token='<PAD>')
    encoder = BertModel.from_pretrained(model_name)

    # apply tokenizer to data and re-align the token indices
    paragraphs = {}
    relations = {}
    labels = {}
    for set_info, raw_data in data.items():
        paragraphs[set_info], relations[set_info], labels[set_info] = prepare(raw_data, tokenizer)

    # model instantiation
    embedding_dim = 768
    model = Model(tokenizer, encoder, embedding_dim, 2)
    if model.use_cuda:
        model.cuda()

    # batchify
    batch_data = batchify(
                    paragraphs['train'],
                    relations['train'],
                    labels['train'],
                    args.batch_size)

    weight = torch.FloatTensor([sum(labels['train'])/(len(labels['train'])-sum(labels['train'])), 1.])
    if model.use_cuda:
        weight = weight.cuda()

    loss_function = nn.NLLLoss(weight=weight, reduction='none')

    if args.feature_only:
        for param in model.encoder.parameters():
            param.requires_grad = False

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in parameters])
    print("#parameters: {}".format(n_params))

    dev_best = 0

    # train the model
    N = len(batch_data)
    for epoch in range(1, args.epochs+1):
        print("Epoch {}:".format(epoch))
        start_time = time.time()
        total_loss = []
        batch_idxs = np.random.permutation(N)
        line_tqdm = tqdm(batch_idxs, dynamic_ncols=True)
        model.train()
        for batch_idx in line_tqdm:

            para, relation, label = batch_data[batch_idx]

            model.zero_grad()

            score = model(para, relation)
            target = torch.LongTensor(label)

            target = Variable(target)
            if model.use_cuda:
                target = target.cuda()

            loss = loss_function(score, target)
            total_loss.extend(loss.data.cpu().numpy().tolist())
            loss.mean().backward()
            optimizer.step()
            gc.collect()
            torch.cuda.empty_cache()

        print("train Loss: {}".format(sum(total_loss)/len(total_loss)))
        end_time = time.time()
        print("Time elapsed: {:.3f}".format(end_time - start_time))

        if args.tensorboard_logging:
            writer.add_histogram("losses", np.asarray(total_loss), epoch, bins='auto')
            writer.add_scalar("TRAIN/loss", sum(total_loss)/len(total_loss), epoch)
        for set_info in ['train', 'dev']:
            print("Test on {} set".format(set_info))

            with torch.no_grad():
                acc, precision, recall, f1, loss = test(model, paragraphs[set_info], relations[set_info], labels[set_info], loss_function)
            if args.tensorboard_logging:
                writer.add_scalar("{}/Accuracy".format(set_info.upper()), acc, epoch)
                writer.add_scalar("{}/Precision".format(set_info.upper()), precision, epoch)
                writer.add_scalar("{}/Recall".format(set_info.upper()), recall, epoch)
                writer.add_scalar("{}/F1".format(set_info.upper()), f1, epoch)
                if set_info == 'dev':
                    writer.add_scalar("{}/Loss".format(set_info.upper()), loss, epoch)

            if set_info == 'dev':
                if f1 > dev_best:
                    print("Save Model...\n")
                    torch.save(model, os.path.join(out_dir, 'bert_best_model.pt'))
                    best_acc = acc
                    best_precision = precision
                    best_recall = recall
                    dev_best = f1

    print("Best Result:")
    print("\tAccuracy: {:0.5f}".format(best_acc))
    print("\tPrecision: {:0.5f}".format(best_precision))
    print("\tRecall: {:0.5f}".format(best_recall))
    print("\tF1: {:0.5f}".format(dev_best))

    return


def model_test(model_file):

    data = load_data('../data/peko_all.jsonl')

    model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name, pad_token='<PAD>')
    paragraphs = {}
    relations = {}
    labels = {}
    for set_info, raw_data in data.items():
        paragraphs[set_info], relations[set_info], labels[set_info] = prepare(raw_data, tokenizer)

    model = torch.load(model_file)

    if model.use_cuda:
        model.cuda()

    model.eval()

    for set_info, dataset in data.items():
        print("Test on {} set".format(set_info))
        with torch.no_grad():
            print("\t#Positive: {}\t#Negative: {}".format(sum(labels[set_info]), len(labels[set_info]) - sum(labels[set_info])))
            pred, acc, precision, recall, f1 = test(model, paragraphs[set_info], relations[set_info], labels[set_info], pred_flag=True)
            with open("{}_with_prediction.jsonl".format(set_info), "w") as fout:
                for instance, p in zip(dataset, pred):
                    instance['prediction'] = p
                    fout.write(json.dumps(instance) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--seed', type=int, default=1234)
    parser.add_argument('-f', '--feature_only', action='store_true')
    parser.add_argument('-ep', '--epochs', type=int, default=50)
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-6)

    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='/home/data/Precondition/')

    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('-ex', '--experiment', type=str, default='test')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.test:
        model_test(args.load_model)
    else:
        train(args)
