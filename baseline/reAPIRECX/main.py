import argparse
from tokenization import PretrainedTokenizer
from pretrain import Trainer

def parse():
    parser = argparse.ArgumentParser(description='APIRecX language model training')

    parser.add_argument('--train_corpus', required=True, type=str, help='subworded corpus for pretraining language model')
    parser.add_argument('--valid_corpus', required=True, type=str, help='subworded corpus for validation in pretrain')
    parser.add_argument('--vocab_file', required=True, type=str, help='pretrained vocabulary')
    parser.add_argument('--pretrained_sp_model', required=True, type=str, help='pretrained sentencepiece model')
    parser.add_argument('--pretrain', action='store_true')
    # parser.add_argument('--finetune', action='store_true')
    # parser.add_argument('--do_eval', action='store_true')

    # parser.add_argument('--test_corpus', default=None, type=str,
                        # help='corpus for either pre-train or fine-tune evaluation')
    parser.add_argument('--models', default=None, type=str, help='pretrained GPT model path')
    # parser.add_argument('--output_model_prefix', default='model', type=str, help='output model name prefix')
    # Input parameters
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--max_seq_len', default=512, type=int, help='the maximum size of the input sequence')
    parser.add_argument('--n_workers', default=8, type=int, help='the number of workers')
    # Train parameters
    parser.add_argument('--epochs', default=15, type=int, help='the number of epochs')
    parser.add_argument('--lr', default=1.5e-4, type=float, help='initial learning rate')
    parser.add_argument('--auxiliary_ratio', default=.25, type=float, help='weight of auxiliary objective')

    # Model parameters
    parser.add_argument('--hidden', default=256, type=int,
                        help='the number of expected features in the transformer decoder')
    parser.add_argument('--n_layers', default=6, type=int, help='the number of decoder layers')
    parser.add_argument('--n_attn_heads', default=8, type=int, help='the number of multi-head attention heads')
    parser.add_argument('--embd_dropout', default=0.1, type=float, help='embedding dropout value')
    parser.add_argument('--resid_dropout', default=0.1, type=float, help='residual dropout value')
    parser.add_argument('--attn_dropout', default=0.1, type=float, help='attention dropout value')
    parser.add_argument('--ffn_hidden', default=512, type=int, help='dimension of the feedforward network')
    # parser.add_argument('--model_output_path', default="gpt_2l_4h", type=str, help='model output path')
    # Others
    # parser.add_argument('--cached_label_dict', default='cached_label_dict.json', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()

    print("init Tokenizer")
    tokenizer = PretrainedTokenizer(pretrained_model=args.pretrained_sp_model, vocab_file=args.vocab_file)

    print("init Trainer")
    trainer = Trainer(args=args,
                      tokenizer=tokenizer)

    if args.pretrain:
        trainer.train()

