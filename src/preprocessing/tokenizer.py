from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class GptTokenizer:
    """
    BPE tokenizer to train the GPT model
    """
    def __init__(self, vocab_size, min_frequency, data_dir):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.data_dir = data_dir

    def train_tokenizer(self):
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=[
                "<UNK>",
                "<PAD>",
                "<SOS>",
                "<EOS>",
            ],
        )

        files = ["../../data/text.txt"]
        tokenizer.train(files, trainer)
        tokenizer.save("tokenizer.json")

    def load_tokenizer(self):
        tokenizer: Tokenizer = Tokenizer.from_file("tokenizer.json")
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<PAD>"), pad_token="<PAD>", length=256)
        return tokenizer
    
tok = GptTokenizer(3, 1, "../../data/")
tok.train_tokenizer()

tokeniz = tok.load_tokenizer()

print(tokeniz.encode("I like to eat apples").ids)

print(tokeniz.encode("I like apples", ).ids)