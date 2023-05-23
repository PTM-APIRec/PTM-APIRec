import argparse
import json

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from data_set import get_test_data_loader
from model import GPTLMHead
from tokenization import PretrainedTokenizer
from tqdm import tqdm

# rejected tokens: control nodes or tokens that cannot be in predicted sub-words
REJECTED_TOKENS = ['[EOS]', '[BOS]', '[PAD]', '[UNK]', 'IF', 'ELSE', 'TRY', 'CATCH', 'FINALLY', 'FOR', 'FOREACH',
                   'ITERABLE', 'VARIABLE', 'WHILE']


class Candidate:
    def __init__(self, ids, pr, is_complete):
        """
        the candidate api in beam search
        :param ids: the id list of subwords generated by beam search
        :param pr: the probability of this candidate
        :param is_complete: whether the candidate api is complete, i.e. has </t> token
        """
        self.ids = ids
        self.pr = pr
        self.is_complete = is_complete


class PredictedAPI:
    def __init__(self, ids, pr):
        """
        the completed candidate api that can be put in the final recommendation list
        :param ids:
        :param pr:
        """
        self.ids = ids
        self.pr = pr


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_seq_len', default=64, type=int, help='the maximum size of the input sequence')
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size')
    parser.add_argument('--vocab_file', default='./vocab/word_12000.vocab', type=str, help='pretrained vocabulary')
    parser.add_argument('--pretrained_sp_model', default='./vocab/word_12000.model', type=str,
                        help='pretrained sentence-piece model')

    parser.add_argument('--beam_size', default=10, type=int, help='beam size in beam search process')
    parser.add_argument('--k', default=10, type=int, help='recommended api list length')
    return parser.parse_args()


def load_model(model_path, device):
    # pretrained_gpt = torch.load(model_path, map_location=torch.device('cuda'))
    pretrained_gpt = torch.load(model_path, map_location=torch.device(device))
    model = GPTLMHead(pretrained_gpt).to(device)
    return model


def test(args, tokenizer, project_name=None):
    # dict: {class: [api...]}, allowed api list for each class
    # class_api_dict = np.load('search_class_api_dict.npy', allow_pickle=True).item()
    # print(class_api_dict)
    beam_size = args.beam_size
    k = args.k

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("use gpu")

    model_path = "./models/pretrained_gpt/gpt_10.pt"
    # load model
    model = load_model(model_path, device)
    model.eval()

    # summary writer
    model_sig = "gpt_10" + "/"
    writer = SummaryWriter("test_logs/" + model_sig)

    # measure
    top_k_num = [0] * 10
    total_num = 0
    mrr = 0.0
    no_class_match_count = 0  # 字典里没有对应的class

    # test
    result_report = []
    test_loader = get_test_data_loader(tokenizer=tokenizer, args=args, project_name=project_name)

    for _, batch in enumerate(test_loader):
        # 按照batch_size等于1，逐个处理
        inputs, target_index = batch
        # print(inputs, target_index)
        input_ids = inputs[0][0].to(device)
        # print("input_ids: ", input_ids, input_ids.shape)
        class_index = inputs[1].to(device)
        # print("class_index: ", class_index, class_index.shape)
        pred_index = inputs[2].to(device)
        # print("pred_index: ", pred_index, pred_index.shape)
        # print("target_index: ", target_index, target_index.shape)

        total_num += pred_index.shape[0]  # total_num + 1

        class_name = tokenizer.convert_ids_to_tokens(input_ids[class_index: pred_index].tolist())
        class_name = "".join(class_name).replace("▁", "").replace("</t>", "")  # 代补全的class名
        # print("class name: ", class_name)
        # if class_name not in class_api_dict.keys():
        #     no_class_match_count += 1
        #     continue

        true_api = tokenizer.convert_ids_to_tokens(input_ids[pred_index + 1: target_index + 1].tolist())
        true_api = "".join(true_api).replace("▁", "").replace("</t>", "")  # 待预测补全的api字符串
        # print(true_api)
        beam_search_list = []  # beam search每步的候选token
        candidate_list = []  # candidate pool: the remaining incomplete chains except the top-k in k^2 results
        generated_api_list = []  # 预测的完整API结果

        over_limit_count = 0  # 预测结果api总序列已经超过规定最大长度

        cur_word = input_ids.contiguous().clone()
        cur_word = cur_word.expand(beam_size, args.max_seq_len)  # [beam_size, 512]
        # print(cur_word)

        # when the smallest pr in generated_api_list is larger than the biggest pr among all incomplete chains
        can_end = False  # the beam search end signal
        beam_iter = 0
        generated_count = 0
        invalid_api_count = 0  # invalid generated api count

        while len(generated_api_list) < k or not can_end:
            if generated_count > 500 or beam_iter > 200:
                # in case the search can't get satisfied result and search forever
                break
            beam_iter += 1
            if len(beam_search_list) > 1:
                # beam search begin
                # print("beam list not null")
                if over_limit_count > 100:
                    # over 100 candidate make api seq over max-seq-len
                    break
                for i in range(len(beam_search_list)):
                    if pred_index + len(beam_search_list[i].ids) >= args.max_seq_len:
                        over_limit_count += 1
                        continue
                    cur_word[i, pred_index + 1:pred_index + 1 + len(beam_search_list[i].ids)] = torch.tensor(
                        beam_search_list[i].ids, dtype=torch.long)

                    current_predict = model(cur_word[i: i + 1, :])
                    # predicted next sub-word
                    single_predicted = current_predict[0, pred_index + len(beam_search_list[i].ids), :].clone()
                    single_predicted = F.softmax(single_predicted, dim=1)
                    predicted_words = torch.argsort(single_predicted, dim=1, descending=True)[0][:beam_size]

                    for next_word in predicted_words:
                        # print("next word", tokenizer.convert_id_to_token(next_word.item()))
                        if tokenizer.convert_id_to_token(next_word.item()).replace("</t>", "") \
                                .replace("▁", "") in REJECTED_TOKENS:
                            continue
                        if tokenizer.convert_id_to_token(next_word.item()).find("</t>") != -1:
                            generated_count += 1
                            update_list = [index for index in beam_search_list[i].ids]
                            update_list.append(next_word.item())
                            generated_api = "".join(tokenizer.convert_ids_to_tokens(update_list)).replace("▁", "") \
                                .replace("</t>", "")

                            # if generated_api not in class_api_dict[class_name]:
                            #     invalid_api_count += 1
                            #     continue
                            # print("add api: ", generated_api)
                            generated_api_list.append(PredictedAPI(update_list,
                                                                   beam_search_list[i].pr
                                                                   * single_predicted[0][next_word.item()].item()))
                            generated_api_list.sort(key=lambda x: x.pr, reverse=True)
                            if len(generated_api_list) > k:
                                generated_api_list.pop(-1)
                        else:
                            update_list = [index for index in beam_search_list[i].ids]
                            update_list.append(next_word.item())
                            candidate_list.append(Candidate(update_list,
                                                            beam_search_list[i].pr *
                                                            single_predicted[0][next_word.item()].item(),
                                                            False))

                candidate_list.sort(key=lambda x: x.pr, reverse=True)

                if len(generated_api_list) > 0 and len(candidate_list) > 0:
                    if candidate_list[0].pr < generated_api_list[-1].pr:
                        can_end = True

                if len(candidate_list) < beam_size:
                    for i in range(len(candidate_list), 0, -1):
                        beam_search_list[i - 1] = candidate_list.pop(i - 1)
                else:
                    for i in range(beam_size, 0, -1):
                        beam_search_list[i - 1] = candidate_list.pop(i - 1)
            else:
                # beam search first step
                current_predict = model(cur_word[0:1, :])
                predicted = current_predict[0, pred_index: pred_index + 1, :]
                predicted = F.softmax(predicted, dim=1)
                predicted_words = torch.argsort(predicted, dim=1, descending=True)[0]
                init_candidate_list, init_generated_list = [], []
                flag_candidate = False  # candidate list长度达到beam size
                # flag_generated = False  # generated list长度达到10, 由于仅靠一个sub-word很难直接生成10个合理api故不考虑
                for i in range(predicted_words.shape[0]):
                    if flag_candidate:
                        break
                    predicted_token = tokenizer.convert_id_to_token(predicted_words[i].item()).replace("▁", "")
                    # print(predicted_token)
                    if len(init_candidate_list) < beam_size:
                        # check whether the word can be first in candidate
                        # api can not start with (, i.e. no api name
                        if predicted_token.find("</t>") == -1 and \
                                (predicted_token.replace("</t>", "") not in REJECTED_TOKENS) and \
                                (not predicted_token.startswith("(")):
                            init_candidate_list.append(
                                Candidate([predicted_words[i].item()],
                                          predicted[0][predicted_words[i].item()].item(),
                                          False))
                    else:
                        flag_candidate = True

                    # if len(init_generated_list) < k:
                    #     if predicted_token.endswith("</t>") and \
                    #             (predicted_token.replace("</t>", "") not in REJECTED_TOKENS) and \
                    #             (not predicted_token.startswith("(")) and (not predicted_token.startswith(")")):
                    #         print("add generated api: ", predicted_token, predicted[0][predicted_words[i]])
                    #         init_generated_list.append(
                    #             PredictedAPI([predicted_words[i].item()],
                    #                          predicted[0][predicted_words[i].item()].item()))
                    # else:
                    #     flag_generated = True

                init_generated_list.sort(key=lambda x: x.pr, reverse=True)

                beam_search_list = [data for data in init_candidate_list]
                candidate_list = [data for data in init_candidate_list]
                # generated_api_list = [data for data in init_generated_list]

                for i in range(beam_size, 0, -1):
                    candidate_list.pop(i - 1)

        # search ends
        # print("beam search ends with beam iter num {}".format(beam_iter))
        # print("generated api list length: ", len(generated_api_list))
        # print("invalid generated api count: ", invalid_api_count)
        generated_api_list.sort(key=lambda x: x.pr, reverse=True)  # 将结果按照可能性从大到小排列
        final_result = []  # 最终要返回的预测结果
        hit_index = -1
        for j in range(len(generated_api_list)):
            result = "".join(tokenizer.convert_ids_to_tokens(generated_api_list[j].ids)) \
                .replace("▁", "").replace("</t>", "")
            if result == true_api:
                hit_index = j
            final_result.append(result)

        item = {"pred": final_result, "true_api": true_api, "hit_index": hit_index}
        result_report.append(item)

        print("predict final results: ", final_result)
        print("predicted hit index: ", hit_index)
        print("Truth: ", true_api)
        if hit_index != -1:
            mrr += 1 / (hit_index + 1)
            for i in range(len(top_k_num)):
                top_k_num[i] += (hit_index <= i)

    print("total num: ", total_num)

    print("top-1: ", top_k_num[0] / total_num)
    print("top-5: ", top_k_num[4] / total_num)
    print("top-10: ", top_k_num[9] / total_num)

    print("mrr: ", mrr / total_num)

    with open("./report.json", "w+") as fp:
        json.dump(result_report, fp=fp, indent=4)

    # -----------log------------
    # top-k, k from 1 to 10
    for i in range(len(top_k_num)):
        top_k_i = top_k_num[i] / total_num
        writer.add_scalar("top-k/" + project_name, top_k_i, i + 1)
    #
    # # mrr
    mrr = mrr / total_num
    writer.add_scalar("mrr/" + project_name, mrr, 0)


if __name__ == "__main__":
    args = parse()

    # 6 test projects
    projects = ["example1"]

    print("init tokenizer")
    tokenizer = PretrainedTokenizer(pretrained_model=args.pretrained_sp_model, vocab_file=args.vocab_file)

    for project_name in projects:
        print("project: ", project_name)
        test(args, tokenizer, project_name)
