import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, tokenizer, encdec, decoding_scheme='beam'):
        super(Model, self).__init__()
        self.use_cuda = True if torch.cuda.is_available() else False
        self.decoding_scheme = decoding_scheme
        self.tokenizer = tokenizer
        self.encdec = encdec
        self.lm_head = nn.Linear(
                encdec.config.n_embd,
                encdec.config.vocab_size,
                bias=False)

    def get_padded_tensor(
            self,
            indexed_tokens,
            target=None,
            event_lens=None,
            weights=None):

        max_len = max([len(ids) for ids in indexed_tokens])
        attention_mask = []
        for i in range(len(indexed_tokens)):
            attention_mask.append(
                    [1]*len(indexed_tokens[i])
                    + [0]*(max_len-len(indexed_tokens[i]))
                    )
            indexed_tokens[i] = indexed_tokens[i] + \
                [self.tokenizer.pad_token_id] * \
                (max_len-len(indexed_tokens[i]))
            if target is not None:
                target[i] = target[i] + [-100] * (max_len-len(target[i]))
                target[i][:event_lens[i]] = [-100] * event_lens[i]
            if weights is not None:
                weights[i] = weights[i] + [1.0] * (max_len-len(weights[i]))

        tokens_tensor = torch.tensor(indexed_tokens)
        attention_mask = torch.tensor(attention_mask)
        if target is not None:
            target = torch.tensor(target)
        if weights is not None:
            weights = torch.tensor(weights)

        if self.use_cuda:
            tokens_tensor = tokens_tensor.cuda()
            attention_mask = attention_mask.cuda()
            if target is not None:
                target = target.cuda()
            if weights is not None:
                weights = weights.cuda()

        return tokens_tensor, attention_mask, target, weights

    def to_list(self, tensor):
        return tensor.cpu().tolist()

    def forward(self, sents, target, event_lens, weights=None):

        pad_result = self.get_padded_tensor(
                            sents,
                            target,
                            event_lens,
                            weights)
        tokens_tensor, attention_mask, target, weights = pad_result

        logits = self.encdec(
                    input_ids=tokens_tensor,
                    attention_mask=attention_mask)[0]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1))

        if weights is not None:
            loss *= weights[..., 1:].contiguous().view(-1)

        return loss.mean()

    def generate(self, sent, eos='<eos>', max_len=30, beam_size=5):
        sent_len = len(sent)
        dec_input = torch.tensor(sent)
        if self.use_cuda:
            dec_input = dec_input.cuda()

        # Generation
        dec_input = dec_input.unsqueeze(0)
        if self.decoding_scheme == 'beam':
            dec_input = self.beam_decoding(
                    dec_input,
                    max_len=max_len,
                    beam_size=beam_size)
        else:
            dec_input = self.greedy_decoding(dec_input)

        # Convert generated seqeunces to string
        gen_sents = [self.tokenizer.decode(s[sent_len:]) for s in dec_input]

        # Filter out invalid generation in terms of markers
        # (<pre>, </pre>, <event>, </event>)
        result = []
        for s in gen_sents:
            sent = []
            event_start_cnt, event_end_cnt = 0, 0
            pre_start_cnt, pre_end_cnt = 0, 0
            event_flag = False
            pre_flag = False
            s_list = s.split()
            for tid, t in enumerate(s_list):
                if t == "<event>":
                    event_start_cnt += 1
                    if tid+2 < len(s_list) and s_list[tid+2] == "</event>":
                        event_flag = True
                if t == "</event>":
                    event_end_cnt += 1
                if t == "<pre>":
                    pre_start_cnt += 1
                    if tid+2 < len(s_list) and s_list[tid+2] == "</pre>":
                        pre_flag = True
                if t == "</pre>":
                    pre_end_cnt += 1

            if event_flag and pre_flag and  \
                    event_start_cnt == 1 and pre_start_cnt == 1 and \
                    event_end_cnt == 1 and pre_end_cnt == 1:
                result.append(s)

        if result == []:
            result = "Failed to generate a precondition"
        else:
            result = result[0]
        return result

    def greedy_decoding(self, dec_input, max_len=20):
        for _ in range(max_len):
            next_hidden = self.encdec(input_ids=dec_input)[0]
            next_token = torch.argmax(
                    F.softmax(next_hidden[:, -1, :], dim=-1), -1)
            dec_input = torch.cat(
                    (dec_input, next_token.view(1, 1)), dim=-1).contiguous()

        return dec_input

    def beam_decoding(self, dec_input, max_len=20, beam_size=5):
        n_candidates = beam_size
        eos_id = self.tokenizer.convert_tokens_to_ids("<eos>")
        scores = torch.FloatTensor([1.0]).view(-1, 1)
        done_sents = []
        done_scores = []
        if self.use_cuda:
            scores = scores.cuda()
        for i in range(max_len):
            next_hidden = self.encdec(input_ids=dec_input)[0]
            prob = F.softmax(
                    next_hidden[:, -1, :], dim=-1)
            scores = scores*prob
            new_input = []

            # Get scores and indices for top k generations
            scores, idxs = torch.topk(scores.view(-1), beam_size)
            scores = scores.view(beam_size, 1)
            idxs = idxs.cpu().tolist()
            seq_idx = []
            vocab_idx = []
            done = []
            not_done = []
            for j, idx in enumerate(idxs):
                seq_idx.append(idx // len(self.tokenizer))
                vocab_idx.append(idx % len(self.tokenizer))
                if (idx % len(self.tokenizer)) == eos_id:
                    done.append(j)
                else:
                    not_done.append(j)

            # Take top k high-scored sequences
            seq_idx = torch.LongTensor(seq_idx)
            vocab_idx = torch.LongTensor(vocab_idx).view(beam_size, 1)
            if self.use_cuda:
                seq_idx = seq_idx.cuda()
                vocab_idx = vocab_idx.cuda()
            new_input = torch.index_select(dec_input, 0, seq_idx)
            dec_input = torch.cat((new_input, vocab_idx), -1)

            # Store if there is any instances reached to <eos> token
            if i > 5 and done:
                done = torch.LongTensor(done)
                if self.use_cuda:
                    done = done.cuda()
                done_tmp = self.to_list(torch.index_select(dec_input, 0, done))
                done_score = [s / (i+1)
                              for s in self.to_list(
                                  -torch.log(
                                      torch.index_select(
                                          scores,
                                          0,
                                          done).view(-1)
                                    ))]

                # Keep sentences in decending order by
                # (Negative Log LM probability) / length
                for k, ss in enumerate(done_score):
                    if done_scores:
                        j = 0
                        while j < len(done_scores):
                            if ss > done_scores[j]:
                                break
                            j += 1
                        if j == len(done_scores):
                            done_scores.append(ss)
                            done_sents.append(done_tmp[k])
                        else:
                            done_scores = done_scores[:j] \
                                + [done_score[k]] + done_scores[j:]
                            done_sents = done_sents[:j] \
                                + [done_tmp[k]] + done_sents[j:]
                    else:
                        done_scores = [ss]
                        done_sents = [done_tmp[k]]

            if len(not_done) == 0:
                break

            # Keep decoding with ones haven't done yet
            not_done = torch.LongTensor(not_done)
            if self.use_cuda:
                not_done = not_done.cuda()
            dec_input = torch.index_select(dec_input, 0, not_done)
            scores = torch.index_select(scores, 0, not_done)
            beam_size = len(dec_input)

        if len(done_sents) < n_candidates:
            done_sents.extend(self.to_list(
                dec_input[:n_candidates - len(done_sents), :]))

        return done_sents
