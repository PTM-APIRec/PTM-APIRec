import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from model import GPT, GPTLMHead
from data_utils import get_data_loaders


class Trainer:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id

        # 预训练语言模型
        if args.pretrained_model:
            self.gpt = torch.load(args.pretrained_model)
        else:
            self.gpt = GPT(vocab_size=self.vocab_size,
                           seq_len=args.max_seq_len,
                           d_model=args.hidden,
                           n_layers=args.n_layers,
                           n_heads=args.n_attn_heads,
                           d_ff=args.ffn_hidden,
                           embd_pdrop=args.embd_dropout,
                           attn_pdrop=args.attn_dropout,
                           resid_pdrop=args.resid_dropout,
                           pad_id=self.pad_id)

        self.model = GPTLMHead(self.gpt)
        self.train_loader, self.valid_loader = get_data_loaders(args, tokenizer)
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
            print("use gpu")

    def train(self):
        print("start training")
        self.model.to(self.device)

        optimizer = optim.RAdam(self.model.parameters(), self.args.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id).to(self.device)

        model_sig = "models/ppt/gpt"
        patience = 0  # 可以接受的验证集准确度没有提升的轮次
        best_acc = 0
        train_loss_list = []

        for epoch in range(self.args.epochs):
            print("start epoch {}".format(epoch))
            losses = 0
            acc = 1
            train_data_num = 1
            n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)
            print(n_batches)
            self.model.train()
            for _, batch in enumerate(self.train_loader):
                inputs = batch[0].to(self.device)
                # print(inputs.shape)
                targets = inputs[:, 1:].contiguous()
                # |inputs| : (batch_size, seq_len), |targets| : (batch_size, seq_len-1)

                lm_logits = self.model(inputs)
                lm_logits = lm_logits[:, :-1].contiguous()
                # |lm_logits| : (batch_size, seq_len-1, vocab_size)
                # acc_1,num = self.compute_acc(targets.view(-1),lm_logits.view(-1, self.vocab_size))
                # acc += acc_1
                # train_data_num += num
                loss = criterion(lm_logits.view(-1, self.vocab_size), targets.view(-1))
                losses += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss_list.append(losses / n_batches)
            print("-----------validating------------")
            valid_acc = self.validate()

            if valid_acc > best_acc:
                print("valid acc VS best acc: ", valid_acc, best_acc)
                best_acc = valid_acc
                patience = 0
            else:
                patience += 1

            print("Train Epoch {} \t Loss: {:.4f}".format(epoch, losses / n_batches))
            print("Epoch {} done, valid accuracy is {}".format(epoch, valid_acc))

            model_path = model_sig + "_%d.pt" % epoch
            print("save model to", model_path)
            torch.save(self.gpt, model_path)
            # torch.save(self.gpt.state_dict(), model_path)

            if patience > 2:
                print("Break at Epoch {}, final loss is {}, best valid acc is {}".format(epoch, losses / n_batches,
                                                                                         best_acc))
                break

        return best_acc

    def validate(self):
        print("start validating")
        valid_acc = 0
        valid_data_num = 0
        self.model.eval()
        with torch.no_grad():
            for _, batch in enumerate(self.valid_loader):
                inputs = batch[0].to(self.device)
                targets = inputs[:, 1:].contiguous()

                lm_logits = self.model(inputs)
                lm_logits = lm_logits[:, :-1].contiguous()

                acc_1, num = self.compute_acc(targets.view(-1), lm_logits.view(-1, self.vocab_size))
                valid_acc += acc_1
                valid_data_num += num

        valid_acc /= valid_data_num
        return valid_acc

    def compute_acc(self, true_tags, logit):
        # prediction = torch.FloatTensor(logit.shape[0], logit.shape[-1]).zero_().to(self.device)
        # print(prediction.dtype)
        select_index = []
        for i in range(logit.shape[0]):
            if true_tags[i].item() != 0:
                select_index.append(i)
        # if (len(select_index)) == 0:
        #     continue
        logit = torch.index_select(logit, 0, torch.tensor(select_index).long().to(self.device))
        true_tags = torch.index_select(true_tags, 0, torch.tensor(select_index).long().to(self.device))
        logit = F.softmax(logit, dim=1)

        # 返回正确的item的数目,eq是返回一个矩阵，sum之后返回总数

        return torch.eq(torch.argmax(logit, dim=1), true_tags).sum().item(), true_tags.shape[0]
