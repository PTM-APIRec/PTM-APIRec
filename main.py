from train import Trainer
from model.PTMAPIRec import PTMAPIRec
from data_prepare.load import *
from data_prepare.api_vocab import ApiVocabulary
from utils import load_model
from test import Test

import os
import argparse


def parse():
    parser = argparse.ArgumentParser("PTMAPIRec")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    parser.add_argument("--model_type", type=str, default="gpt")
    parser.add_argument("--pre_trained_tokenizer", default="pretrained_model/CodeGPT-small-java-adaptedGPT2")
    parser.add_argument("--pre_trained_model", default="pretrained_model/CodeGPT-small-java-adaptedGPT2")
    parser.add_argument("--vocab_file", default="data/api_vocab/android_api.json")

    parser.add_argument("--train_dataset", default="data/dataset/android/train.json")
    parser.add_argument("--valid_dataset", default="data/dataset/android/valid.json")
    parser.add_argument("--test_dataset", default="data/dataset/android/")

    parser.add_argument("--model_sig", default="ptmapirec")
    parser.add_argument("--model_dir", default="./train_model", help="Directory of output model weight.")
    parser.add_argument("--log_dir", default="./train_logs")
    parser.add_argument("--test_dir", default="./test_logs")
    parser.add_argument("--stat_dir", default=None)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--api_dim", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--dropout", default=0.3)
    parser.add_argument("--lr", default=1e-3)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--saved_model", default=None)
    parser.add_argument("--saved_model_dir", default=None)

    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--random_init_model", action="store_true")
    parser.add_argument("--require_signature", action="store_true")
    parser.add_argument("--ignore_pt", action="store_true")

    return parser.parse_args()


def main():
    args = parse()
    api_vocab = ApiVocabulary.from_json(args.vocab_file)
    model_type = args.model_type
    pretrained_tokenizer = load_pre_trained_tokenizer(args.pre_trained_tokenizer)
    pretrained_model = load_pre_trained_model(args.pre_trained_model,
                                              config_only=args.random_init_model,
                                              model_type=model_type)

    model = PTMAPIRec(pre_trained_model=pretrained_model,
                      hidden_size=args.hidden_size,
                      api_size=len(api_vocab),
                      api_dim=args.api_dim,
                      dropout=args.dropout,
                      model_type=args.model_type)
    if args.ignore_pt:
        print(f"Freeze transformer model.")
        for name, param in model.named_parameters():
            if name.startswith("transformer") \
                    or name.startswith("encoder") \
                    or name.startswith("decoder") \
                    or name.startswith("shared"):
                param.requires_grad = False

    if args.train:
        train_path = args.train_dataset
        if not args.wo_vocab:
            train_ds, valid_ds = load_model_dataset(train_path,
                                                    pretrained_tokenizer,
                                                    args.batch_size,
                                                    max_token_len=args.max_seq_len,
                                                    model_type=model_type
                                                    )
        else:
            train_ds, valid_ds = load_wo_data(train_path,
                                              pretrained_tokenizer,
                                              args.batch_size,
                                              max_token_len=args.max_seq_len,
                                              )
        trainer = Trainer(model=model,
                          api_vocabs=api_vocab,
                          train_dataset=train_ds,
                          valid_dataset=valid_ds,
                          args=args)
        trainer.train()
    elif args.test:
        saved_model_dir = args.saved_model_dir
        saved_model = args.saved_model
        test_path = args.test_dataset
        test_name = ["test_long", "test_normal", "test_short"]
        test_datasets = {
            tn:
                load_model_dataset(os.path.join(test_path, tn + ".json"),
                                   pretrained_tokenizer, args.batch_size,
                                   max_token_len=args.max_seq_len,
                                   model_type=model_type,
                                   test=True)
            for tn in test_name
        }
        if saved_model_dir is not None:
            for pt_file in os.listdir(saved_model_dir):
                filename = os.fsdecode(pt_file)
                if filename.endswith(".pt"):
                    print("Start testing {}.".format(filename))
                    test_model_name = filename[:-3]
                    model_path = os.path.join(saved_model_dir, filename)
                    model = load_model(model, model_path)
                    tester = Test(test_datasets, model, args, model_name=test_model_name)
                    tester.test(stat_dir=args.stat_dir)
        elif saved_model is not None:
            filename = os.path.basename(saved_model)
            test_model_name = filename[:-3]
            model = load_model(model, saved_model)
            tester = Test(test_datasets, model, args, model_name=test_model_name)
            tester.test(stat_dir=args.stat_dir)


if __name__ == '__main__':
    main()
