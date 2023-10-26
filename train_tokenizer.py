if __name__ == "__main__":
    import argparse
    import json
    from tokenizers.models import BPE
    from tokenizers import Tokenizer
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from hw_asr.base.base_text_encoder import BaseTextEncoder

    parser = argparse.ArgumentParser(
                prog='BPE trainer',
                description='Use me to train BPE tokenizer')
    parser.add_argument("--unk", default="<unk>")
    parser.add_argument("--pad", default="<pad>")
    parser.add_argument("--max_token_length", default=6, type=int)
    parser.add_argument("--vocab_size", default=800, type=int)
    parser.add_argument("--out")
    parser.add_argument("input", metavar="I", nargs='+',
                        help='dataset indexes to train on')
    args = parser.parse_args()
    tokenizer = Tokenizer(BPE(unk_token=args.unk)) # pad_token_id=0, pad_token=args.pad
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens =[args.pad, args.unk], 
                         vocab_size=args.vocab_size, 
                         max_token_length=args.max_token_length)
    corpus = [['_ '] * 100]
    for path in args.input:
        with open(path) as file:
            j = json.load(file)
            for el in j:
                corpus.append(BaseTextEncoder.normalize_text(el['text']))
    tokenizer.train_from_iterator(corpus, trainer) # training the tokenzier
    tokenizer.save(args.out)