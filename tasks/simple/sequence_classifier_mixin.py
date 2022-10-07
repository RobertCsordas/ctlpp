import torch
import torch.nn
import torch.utils.data
from models import SequenceClassifier
from interfaces import SequenceClassifierInterface
from .. import args
import framework
from layers import CudaLSTM
from models.encoder_decoder import Encoder, BidirectionalLSTMEncoder


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-seq_classifier.rnn", default="bilstm", choice=["bilstm"])
    parser.add_argument("-thinking_steps", default=0)


class SequenceClassifierMixin:
    def create_model(self) -> torch.nn.Module:
        rnns = {
            "bilstm": CudaLSTM,
        }

        encoders = {
            "bilstm": BidirectionalLSTMEncoder
        }

        model = SequenceClassifier(len(self.train_set.in_vocabulary),
                                   len(self.train_set.out_vocabulary), self.helper.args.state_size,
                                   self.helper.args.n_layers,
                                   self.helper.args.embedding_size,
                                   self.helper.args.dropout,
                                   lstm = rnns.get(self.helper.args.seq_classifier.rnn),
                                   encoder = encoders.get(self.helper.args.seq_classifier.rnn, Encoder),
                                   n_thinking_steps=self.helper.args.thinking_steps)

        self.n_weights = sum(p.numel() for p in model.parameters())
        return model

    def create_model_interface(self):
        self.model_interface = SequenceClassifierInterface(self.model)
