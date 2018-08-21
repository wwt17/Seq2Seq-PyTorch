import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.outdim = resnet.fc.in_features
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
            features = features.reshape(features.size(0), -1)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        
        self.init_linear = nn.Linear(feature_size, embed_size)
        self.init_bn = nn.BatchNorm1d(embed_size, momentum=0.01)

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions, states=None,
                max_decode_length=None, beam=None, teach_flags=None):
        features = self.init_bn(self.init_linear(features))
        tgt = self.embed(captions)

        if max_decode_length is None:
            max_decode_length = captions.shape[1]

        if teach_flags is None:
            teach_flags = [True] + [False] * max_decode_length

        if beam is None:
            get_next = lambda logit, tgt, step: tgt[:, step]
        elif beam == 0:
            get_next = lambda logit, tgt, step: (tgt[:, step] if teach_flags[step] else torch.mm(F.softmax(logit, -1), self.embed.weight))
        elif beam > 0:
            get_next = lambda logit, tgt, step: (tgt[:, step] if teach_flags[step] else self.embed(logit.max(-1)[1]))

        logits = []
        for step in range(-1, max_decode_length):
            input = get_next(logit, tgt, step) if step > -1 else features
            output, states = self.lstm(input.unsqueeze(1), states)
            output = output.squeeze(1)
            logit = self.linear(output)
            if step > -1:
                logits.append(logit)

        logits = torch.stack(logits, 1)
        return logits
