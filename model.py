import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    """This is a neural network architecture for an encoder module in a image captioning model, implemented using PyTorch's nn module.
    It has two main parts: a pre-trained ResNet-50 model and an embedding linear layer.
    The pre-trained ResNet-50 model (resnet) is a deep convolutional neural network trained on ImageNet dataset. 
    It is used to extract features from the input images. The parameters of this model are frozen, meaning they will not be updated during training.
    The embedding linear layer (self.embedding_layer) is used to project the features extracted by the ResNet-50 model to a lower-dimensional space of size embed_size.
    The input of this network is an image and the output is a lower-dimensional feature representation of the image.
    It uses the resnet50 pre-trained model to extract features, which is fine-tuned by the last linear layer for the specific task of image captioning.
"""
    
    def __init__(self, embed_size):
        super(ImageEncoder, self).__init__()
        resnet = models.resnet101(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embedding_layer = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embedding_layer(features)
        return features




def weights_init(m):
    '''
    Function to initialize weights of the model with xavier initialization.
    '''
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    
class TextDecoder(nn.Module):
    """This is a neural network architecture for a decoder module in an image captioning model, implemented using PyTorch's nn module. It has four main parts: an embedding layer, an LSTM layer, a fully connected output layer and a weight initialization function.

The embedding layer (self.embedding_layer) is used to convert the input word indices to dense vectors of fixed size (embed_size).
The LSTM layer (self.lstm) is a type of recurrent neural network that is used for processing sequential data. It has hidden_size number of hidden states, dropout of 0.5 and num_layers number of layers.
The fully connected output layer (self.fc_out) is used to generate the final output of the network, which has vocab_size number of outputs.
The weights_init() function initializes the weights of the network.
The forward method takes as input both image features, and captions, where captions are passed to the embedding layer, and then concatenated with the image features. 
The concatenated features are then passed through the LSTM layers and the final output is generated by passing the output of the LSTM layer through the fully connected output layer. 
The output of this network is a probability distribution over the vocabulary.
It also has a device setup to perform the operations on either CPU or GPU, based on the availability.
It also initializes hidden and cell states of LSTM to zero before passing the input through LSTM layers.
"""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(TextDecoder, self).__init__()
        
        # sizes of the model's blocks
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        
        # embedding layer
        self.embedding_layer = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        
        # lstm unit(s)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first = True, dropout = 0.35, num_layers = self.num_layers)
    
        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        
        # initialize the weights
        self = self.apply(weights_init)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        
        # setup the device
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        # batch size
        batch_size = features.size(0)
        
        # init the hidden and cell states to zeros
        self.hidden_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        self.cell_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)

        # embedding_layer the captions
        captions_embed = self.embedding_layer(captions)
        
        # pass through lstm unit(s)
        vals = torch.cat((features.unsqueeze(1), captions_embed), dim=1)
        outputs, (self.hidden_state, self.cell_state) = self.lstm(vals, (self.hidden_state, self.cell_state))
        
        # pass through the linear unit
        outputs = self.fc_out(outputs)
            
        return outputs

    def sample(self, inputs, hidden_size, states=None, max_len=20):
        output = []
        batch_size = inputs.shape[0]
        hidden = (torch.randn(1, 1, hidden_size).to(inputs.device),
              torch.randn(1, 1, hidden_size).to(inputs.device))

        while True:
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.fc_out(lstm_out)
            outputs = outputs.squeeze(1)
            _, max_pred_index = torch.max(outputs, dim = 1)
            output.append(max_pred_index.cpu().numpy()[0].item())
            if (max_pred_index == 1):
                break
            inputs = self.embedding_layer(max_pred_index)
            inputs = inputs.unsqueeze(1)
            
        return output