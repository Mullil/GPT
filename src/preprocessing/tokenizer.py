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

        files = ["corpus.txt"]
        tokenizer.train(files, trainer)
        tokenizer.save("tokenizer.json")

    def load_tokenizer(self):
        tokenizer = Tokenizer.from_file("tokenizer.json")
        return tokenizer
    
tok = GptTokenizer(3, 1, "aa")
tok.train_tokenizer()