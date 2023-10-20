dataset: 'dataset/' # dataset directory
models: 'models/' # for storing models
plots: 'plots/' # for plots
pickle: 'pickle/' # pickle dumps root path

cuda: True #whether to use NVIDIA cuda
    
test_per_epoch: 0 #test per epoch i.e. how many times in ONE epoch
test_every_epoch: 50 #after how many epochs
print_per_epoch: 3 #print loss function how often in each epoch
save_every: 400 #save models after how many epochs
plot_every: 25 #save plots for test loss/train loss/accuracy

resume: False #resume training from saved model

lr: 0.0005 #learning rate

drop_uniform: False #whether dropping of character sets is independent of set size
reset_after: 400 #generate a new random dataset after these manh epochs
vocab_size: 26 #size of vocabulary. 26 engliush letters in our case
min_len: 3 #words with length less than min_len are not added to the dataset

# rnn: 'GRU' #type of RNN. Can be LSTM/GRU
use_embedding: True #whether to use character embeddings
embedding_dim: 128 #if use_embedding, dimension of embedding
hidden_dim: 512 #hidden dimension of RNN
num_head: 8
output_mid_features: 256 #number of neurons in hidden layer after RNN
miss_linear_dim: 256 #miss chars are projected to this dimension using a simple linear layer
num_layers: 3 #number of layers in RNN
dropout: 0.1 #dropout
batch_size: 4000 #batch size for training and testing
epochs: 3000 #total no. of epochs to train

"""
The main driver file responsible for training, testing and predicting
"""
import sys
sys.path.append('')


import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import pickle
from model import Transformer
from dataloader import dataloader
from dataloader import encoded_to_string

#load config file
with open("config-trans.yaml", 'r') as stream:
	try:
		config = yaml.safe_load(stream)
	except yaml.YAMLError as exc:
		print(exc)

#class responsible for training, testing and inference
class dl_model():

	def __init__(self, mode):

		# Read config fielewhich contains parameters
		self.config = config
		self.mode = mode

		# Architecture name decides prefix for storing models and plots
		feature_dim = self.config['vocab_size']
		self.arch_name = '_'.join(
			['Transformer', str(self.config['num_layers']), str(self.config['hidden_dim']), str(feature_dim)])

		print("Architecture:", self.arch_name)
		# Change paths for storing models
		self.config['models'] = self.config['models'].split('/')[0] + '_' + self.arch_name + '/'
		self.config['plots'] = self.config['plots'].split('/')[0] + '_' + self.arch_name + '/'

		# Make folders if DNE
		if not os.path.exists(self.config['models']):
			os.mkdir(self.config['models'])
		if not os.path.exists(self.config['plots']):
			os.mkdir(self.config['plots'])
		if not os.path.exists(self.config['pickle']):
			os.mkdir(self.config['pickle'])

		self.cuda = (self.config['cuda'] and torch.cuda.is_available())

		# load/initialise metrics to be stored and load model
		if mode == 'train' or mode == 'test':

			self.plots_dir = self.config['plots']
			# store hyperparameters
			self.total_epochs = self.config['epochs']
			self.test_every = self.config['test_every_epoch']
			self.test_per = self.config['test_per_epoch']
			self.print_per = self.config['print_per_epoch']
			self.save_every = self.config['save_every']
			self.plot_every = self.config['plot_every']

			# dataloader which returns batches of data
			self.train_loader = dataloader('train', self.config)
			self.test_loader = dataloader('test', self.config)
			# declare model
			self.model = Transformer(self.config)

			self.start_epoch = 1
			self.edit_dist = []
			self.train_losses, self.test_losses = [], []

		else:

			self.model = Transformer(self.config)

		if self.cuda:
			self.model.cuda()

		# resume training from some stored model
		if self.mode == 'train' and self.config['resume']:
			self.start_epoch, self.train_losses, self.test_losses = self.model.load_model(mode, self.model.rnn_name, self.model.num_layers, self.model.hidden_dim)
			self.start_epoch += 1

		# load best model for testing/inference
		elif self.mode == 'test' or mode == 'test_one':
			self.model.load_model(mode, self.config['rnn'], self.model.num_layers, self.model.hidden_dim)

		#whether using embeddings
		if self.config['use_embedding']:
			self.use_embedding = True
		else:
			self.use_embedding = False

	# Train the model
	def train(self):

		print("Starting training at t =", datetime.datetime.now())
		print('Batches per epoch:', len(self.train_loader))
		self.model.train()

		# when to print losses during the epoch
		print_range = list(np.linspace(0, len(self.train_loader), self.print_per + 2, dtype=np.uint32)[1:-1])
		if self.test_per == 0:
			test_range = []
		else:
			test_range = list(np.linspace(0, len(self.train_loader), self.test_per + 2, dtype=np.uint32)[1:-1])

		for epoch in range(self.start_epoch, self.total_epochs + 1):

			try:

				print("Epoch:", str(epoch))
				epoch_loss = 0.0
				# i used for monitoring batch and printing loss, etc.
				i = 0

				while True:

					i += 1

					# Get batch of inputs, labels, missed_chars and lengths along with status (when to end epoch)
					inputs, labels, miss_chars, input_lens, status = self.train_loader.return_batch()

					if self.use_embedding:
						inputs = torch.from_numpy(inputs).long() #embeddings should be of dtype long
					else:
						inputs = torch.from_numpy(inputs).float()

					#convert to torch tensors
					labels = torch.from_numpy(labels).float()
					miss_chars = torch.from_numpy(miss_chars).float()
					input_lens = torch.from_numpy(input_lens).long()

					if self.cuda:
						inputs = inputs.cuda()
						labels = labels.cuda()
						miss_chars = miss_chars.cuda()
						input_lens = input_lens.cuda()

					# zero the parameter gradients
					self.model.optimizer.zero_grad()
					# forward + backward + optimize
					outputs = self.model(inputs, input_lens, miss_chars)
					loss, miss_penalty = self.model.calculate_loss(outputs, labels, input_lens, miss_chars, self.cuda)
					loss.backward()

					# clip gradient
					# torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
					self.model.optimizer.step()

					# store loss
					epoch_loss += loss.item()

					# print loss
					if i in print_range and epoch == 1:
						print('After %i batches, Current Loss = %.7f' % (i, epoch_loss / i))
					elif i in print_range and epoch > 1:
						print('After %i batches, Current Loss = %.7f, Avg. Loss = %.7f, Miss Loss = %.7f' % (
								i, epoch_loss / i, np.mean(np.array([x[0] for x in self.train_losses])), miss_penalty))

					# test model periodically
					if i in test_range:
						self.test(epoch)
						self.model.train()

					# Reached end of dataset
					if status == 1:
						break

				#refresh dataset i.e. generate a new dataset from corpurs
				if epoch % self.config['reset_after'] == 0:
					self.train_loader.refresh_data(epoch)

				#take the last example from the epoch and print the incomplete word, target characters and missed characters
				random_eg = min(np.random.randint(self.train_loader.batch_size), inputs.shape[0]-1)
				encoded_to_string(inputs.cpu().numpy()[random_eg], labels.cpu().numpy()[random_eg], miss_chars.cpu().numpy()[random_eg],
								  input_lens.cpu().numpy()[random_eg], self.train_loader.char_to_id, self.use_embedding)

				# Store tuple of training loss and epoch number
				self.train_losses.append((epoch_loss / len(self.train_loader), epoch))

				# save model
				if epoch % self.save_every == 0:
					self.model.save_model(False, epoch, self.train_losses, self.test_losses,
										  self.model.rnn_name, self.model.num_layers, self.model.hidden_dim)

				# test every 5 epochs in the beginning and then every fixed no of epochs specified in config file
				# useful to see how loss stabilises in the beginning
				if epoch % 5 == 0 and epoch < self.test_every:
					self.test(epoch)
					self.model.train()
				elif epoch % self.test_every == 0:
					self.test(epoch)
					self.model.train()
				# plot loss and accuracy
				if epoch % self.plot_every == 0:
					self.plot_loss_acc(epoch)

			except KeyboardInterrupt:
				#save model before exiting
				print("Saving model before quitting")
				self.model.save_model(False, epoch-1, self.train_losses, self.test_losses,
									  self.model.rnn_name, self.model.num_layers, self.model.hidden_dim)
				exit(0)


	# test model
	def test(self, epoch=None):

		self.model.eval()

		print("Testing...")
		print('Total batches:', len(self.test_loader))
		test_loss = 0

		#generate a new dataset form corpus
		self.test_loader.refresh_data(epoch)

		with torch.no_grad():

			while True:

				# Get batch of input, labels, missed characters and lengths along with status (when to end epoch)
				inputs, labels, miss_chars, input_lens, status = self.test_loader.return_batch()
				
				if self.use_embedding:
					inputs = torch.from_numpy(inputs).long()
				else:
					inputs = torch.from_numpy(inputs).float()

				labels = torch.from_numpy(labels).float()
				miss_chars = torch.from_numpy(miss_chars).float()
				input_lens= torch.from_numpy(input_lens).long()

				if self.cuda:
					inputs = inputs.cuda()
					labels = labels.cuda()
					miss_chars = miss_chars.cuda()
					input_lens = input_lens.cuda()

				# zero the parameter gradients
				self.model.optimizer.zero_grad()
				# forward + backward + optimize
				outputs = self.model(inputs, input_lens, miss_chars)
				loss, miss_penalty = self.model.calculate_loss(outputs, labels, input_lens, miss_chars, self.cuda)
				test_loss += loss.item()

				# Reached end of dataset
				if status == 1:
					break

		#take a random example from the epoch and print the incomplete word, target characters and missed characters
		#min since the last batch may not be of length batch_size
		random_eg = min(np.random.randint(self.train_loader.batch_size), inputs.shape[0]-1)
		encoded_to_string(inputs.cpu().numpy()[random_eg], labels.cpu().numpy()[random_eg], miss_chars.cpu().numpy()[random_eg],
			input_lens.cpu().numpy()[random_eg], self.train_loader.char_to_id, self.use_embedding)

		# Average out the losses and edit distance
		test_loss /= len(self.test_loader)

		print("Test Loss: %.7f, Miss Penalty: %.7f" % (test_loss, miss_penalty))

		# Store in lists for keeping track of model performance
		self.test_losses.append((test_loss, epoch))

		# if testing loss is minimum, store it as the 'best.pth' model, which is used during inference
		# store only when doing train/test together i.e. mode is train
		if test_loss == min([x[0] for x in self.test_losses]) and self.mode == 'train':
			print("Best new model found!")
			self.model.save_model(True, epoch, self.train_losses, self.test_losses,
								  self.model.rnn_name, self.model.num_layers, self.model.hidden_dim)

		return test_loss

	def predict(self, string, misses, char_to_id):
		"""
		called during inference
		:param string: word with predicted characters and blanks at remaining places
		:param misses: list of characters which were predicted but game feedback indicated that they are not present
		:param char_to_id: mapping from characters to id
		"""

		id_to_char = {v:k for k,v in char_to_id.items()}

		#convert string into desired input tensor
		if self.use_embedding:
			encoded = np.zeros((len(char_to_id)))
			for i, c in enumerate(string):
				if c == '.':
					encoded[i] = len(id_to_char) - 1 
				else:
					encoded[i] = char_to_id[c]

			inputs = np.array(encoded)[None, :]
			inputs = torch.from_numpy(inputs).long()

		else:

			encoded = np.zeros((len(string), len(char_to_id)))
			for i, c in enumerate(string):
				if c == '.':
					encoded[i][len(id_to_char) - 1] = 1
				else:
					encoded[i][char_to_id[c]] = 1

			inputs = np.array(encoded)[None, :, :]
			inputs = torch.from_numpy(inputs).float()

		#encode the missed characters
		miss_encoded = np.zeros((len(char_to_id) - 1))
		for c in misses:
			miss_encoded[char_to_id[c]] = 1
		miss_encoded = np.array(miss_encoded)[None, :]
		miss_encoded = torch.from_numpy(miss_encoded).float()

		input_lens = np.array([len(string)])
		input_lens= torch.from_numpy(input_lens).long()	

		#pass through model
		output = self.model(inputs, input_lens, miss_encoded).detach().cpu().numpy()[0]

		#sort predictions
		sorted_predictions = np.argsort(output)[::-1]
		
		#we cannnot consider only the argmax since a missed character may also get assigned a high probability
		#in case of a well-trained model, we shouldn't observe this
		return [id_to_char[x] for x in sorted_predictions]

	def plot_loss_acc(self, epoch):
		"""
		take train/test loss and test accuracy input and plot it over time
		:param epoch: to track performance across epochs
		"""

		plt.clf()
		fig, ax1 = plt.subplots()

		ax1.set_xlabel('Epoch')
		ax1.set_ylabel('Loss')
		ax1.plot([x[1] for x in self.train_losses], [x[0] for x in self.train_losses], color='r', label='Train Loss')
		ax1.plot([x[1] for x in self.test_losses], [x[0] for x in self.test_losses], color='b', label='Test Loss')
		ax1.tick_params(axis='y')
		ax1.legend(loc='upper left')

		fig.tight_layout()  # otherwise the right y-label is slightly clipped
		plt.grid(True)
		plt.legend()
		plt.title(self.arch_name)

		filename = self.plots_dir + 'plot_' + self.arch_name + '_' + str(epoch) + '.png'
		plt.savefig(filename)

		print("Saved plots")


if __name__ == '__main__':

	a = dl_model('train')
	a.train()
	# char_to_id = {chr(97+x): x+1 for x in range(26)}
	# char_to_id['PAD'] = 0
	# a = dl_model('test_one')
	# print(a.predict(".oau", char_to_id))


import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from generic_model import generic_model
from embed import TransformerEncoder

# generic model contains generic methods for loading and storing a model
class RNN(generic_model):

    def __init__(self, config):

        super(RNN, self).__init__(config)

        # Store important parameters
        self.rnn_name = config['rnn']
        self.input_dim = config['vocab_size'] + 1
        self.hidden_dim = config['hidden_dim'] 
        self.num_layers = config['num_layers']
        self.embed_dim = config['embedding_dim']
        self.output_dim = config['vocab_size']

        #whether to use character embeddings
        if config['use_embedding']:
            self.use_embedding = True
            self.embedding = nn.Embedding(self.input_dim, self.embed_dim)
        else:
            self.use_embedding = False
            
        #linear layer after RNN output
        in_features = config['miss_linear_dim'] + self.hidden_dim*2
        mid_features = config['output_mid_features']
        self.linear1_out = nn.Linear(in_features, mid_features)
        self.relu = nn.ReLU()
        self.linear2_out = nn.Linear(mid_features, self.output_dim)

        #linear layer after missed characters
        self.miss_linear = nn.Linear(config['vocab_size'], config['miss_linear_dim'])        

        #declare RNN
        if self.rnn_name == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.embed_dim if self.use_embedding else self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                               dropout=config['dropout'],
                               bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=self.embed_dim if self.use_embedding else self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                              dropout=config['dropout'],
                              bidirectional=True, batch_first=True)

        #optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])

    def forward(self, x, x_lens, miss_chars):
        """
        Forward pass through RNN
        :param x: input tensor of shape (batch size, max sequence length, input_dim)
        :param x_lens: actual lengths of each sequence < max sequence length (since padded with zeros)
        :param miss_chars: tensor of length batch_size x vocab size. 1 at index i indicates that ith character is NOT present
        :return: tensor of shape (batch size, max sequence length, output dim)
        """        
        if self.use_embedding:
            x = self.embedding(x)
            
        batch_size, seq_len, _ = x.size()
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        # now run through RNN
        output, hidden = self.rnn(x)
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = hidden[-1]
        hidden = hidden.permute(1, 0, 2)

        hidden = hidden.contiguous().view(hidden.shape[0], -1)
        #project miss_chars onto a higher dimension
        miss_chars = self.miss_linear(miss_chars)
        #concatenate RNN output and miss chars
        concatenated = torch.cat((hidden, miss_chars), dim=1)
        #predict
        return self.linear2_out(self.relu(self.linear1_out(concatenated)))

    def calculate_loss(self, model_out, labels, input_lens, miss_chars, use_cuda):
        """
        :param model_out: tensor of shape (batch size, max sequence length, output dim) from forward pass
        :param labels: tensor of shape (batch size, vocab_size). 1 at index i indicates that ith character should be predicted
        :param: miss_chars: tensor of length batch_size x vocab size. 1 at index i indicates that ith character is NOT present
							passed here to check if model's output probability of missed_chars is decreasing
        """
        outputs = nn.functional.log_softmax(model_out, dim=1)
        #calculate model output loss for miss characters
        miss_penalty = torch.sum(outputs*miss_chars, dim=(0,1))/outputs.shape[0]
        
        input_lens = input_lens.float()
        #weights per example is inversely proportional to length of word
        #this is because shorter words are harder to predict due to higher chances of missing a character
        weights_orig = (1/input_lens)/torch.sum(1/input_lens).unsqueeze(-1)
        weights = torch.zeros((weights_orig.shape[0], 1))    
        #resize so that torch can process it correctly
        weights[:, 0] = weights_orig

        if use_cuda:
        	weights = weights.cuda()
        
        #actual loss
        loss_func = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
        actual_penalty = loss_func(model_out, labels)
        return actual_penalty, miss_penalty
        
# Transformer: generic model contains generic methods for loading and storing a model
class Transformer(generic_model):

    def __init__(self, config):

        super(Transformer, self).__init__(config)

        # Store important parameters
        self.input_dim = config['vocab_size'] + 1
        self.hidden_dim = config['hidden_dim'] 
        self.output_dim = config['vocab_size']
        self.num_layers = config['num_head']

        #whether to use character embeddings
        if config['use_embedding']:
            self.use_embedding = True
            self.embedding = nn.Embedding(self.input_dim, self.hidden_dim)
        else:
            self.use_embedding = False
            
        #linear layer after Transformer output
        in_features = config['miss_linear_dim'] + self.hidden_dim*2
        mid_features = config['output_mid_features']
        self.linear1_out = nn.Linear(in_features, mid_features)
        self.relu = nn.ReLU()
        self.linear2_out = nn.Linear(mid_features, self.output_dim)

        #linear layer after missed characters
        self.miss_linear = nn.Linear(config['vocab_size'], config['miss_linear_dim'])        
        # declare transformer           
        self.encoder = TransformerEncoder(n_layers=self.num_layers, vocab_size=self.input_dim, embed_dim=self.hidden_dim)

        #optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])

    def forward(self, x, x_lens, miss_chars):
        """
        Forward pass through Transformer
        :param x: input tensor of shape (batch size, max sequence length, input_dim)
        :param x_lens: actual lengths of each sequence < max sequence length (since padded with zeros)
        :param miss_chars: tensor of length batch_size x vocab size. 1 at index i indicates that ith character is NOT present
        :return: tensor of shape (batch size, max sequence length, output dim)
        """        
        if self.use_embedding:
            x = self.embedding(x)
            
        batch_size, seq_len, _ = x.size()
        # x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        
        # now run through Transformer Encoder
        output = self.encoder(x)
        print(output.shape)
        # hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        # hidden = hidden[-1]
        # hidden = hidden.permute(1, 0, 2)

        # hidden = hidden.contiguous().view(hidden.shape[0], -1)
        #project miss_chars onto a higher dimension
        miss_chars = self.miss_linear(miss_chars)
        #concatenate RNN output and miss chars
        concatenated = torch.cat((output, miss_chars), dim=1)
        #predict
        return self.linear2_out(self.relu(self.linear1_out(concatenated)))

    def calculate_loss(self, model_out, labels, input_lens, miss_chars, use_cuda):
        """
        :param model_out: tensor of shape (batch size, max sequence length, output dim) from forward pass
        :param labels: tensor of shape (batch size, vocab_size). 1 at index i indicates that ith character should be predicted
        :param: miss_chars: tensor of length batch_size x vocab size. 1 at index i indicates that ith character is NOT present
							passed here to check if model's output probability of missed_chars is decreasing
        """
        outputs = nn.functional.log_softmax(model_out, dim=1)
        #calculate model output loss for miss characters
        miss_penalty = torch.sum(outputs*miss_chars, dim=(0,1))/outputs.shape[0]
        
        input_lens = input_lens.float()
        #weights per example is inversely proportional to length of word
        #this is because shorter words are harder to predict due to higher chances of missing a character
        weights_orig = (1/input_lens)/torch.sum(1/input_lens).unsqueeze(-1)
        weights = torch.zeros((weights_orig.shape[0], 1))    
        #resize so that torch can process it correctly
        weights[:, 0] = weights_orig

        if use_cuda:
        	weights = weights.cuda()
        
        #actual loss
        loss_func = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
        actual_penalty = loss_func(model_out, labels)
        return actual_penalty, miss_penalty
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, dropout=0, max_len=1000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.pe = nn.Parameter(pe, requires_grad=False)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

def scaled_dot_product_attention(query, key, value, query_mask=None, key_mask=None, mask=None):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    if query_mask is not None and key_mask is not None:
        mask = torch.bmm(query_mask.unsqueeze(-1), key_mask.unsqueeze(1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -float("inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        attn_outputs = scaled_dot_product_attention(
            self.q(query), self.k(key), self.v(value), query_mask, key_mask, mask)
        return attn_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, query_mask=None, key_mask=None, mask=None):
        x = torch.cat([
            h(query, key, value, query_mask, key_mask, mask) for h in self.heads
        ], dim=-1)
        x = self.output_linear(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_dim, middle_dim, drop_prob):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, middle_dim)
        self.linear_2 = nn.Linear(middle_dim, embed_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, middle_dim, drop_prob):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, middle_dim, drop_prob)

    def forward(self, x, mask=None):
        # Apply layer normalization and then copy input into query, key, value
        hidden_state = self.layer_norm_1(x)
        # Apply attention with a skip connection
        x = x + self.attention(hidden_state, hidden_state, hidden_state, mask=mask)
        # Apply feed-forward layer with a skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        X2 = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens-1, 2, dtype=torch.float32) / (num_hiddens-1))
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X2)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return X

class Embeddings(nn.Module):
    def __init__(self, ori_feature_dim, embed_dim, drop_prob):
        super().__init__()
        # self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.token_embeddings = nn.Linear(ori_feature_dim, embed_dim) # change embedding into linear for onehot
        self.layer_norm = nn.LayerNorm(embed_dim, eps=1e-12)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, inputs):
        #inputs = torch.tensor(inputs).to(inputs.device).long()
        # Create token and position embeddings
        token_embeddings = self.token_embeddings(inputs)
        PE = PositionalEncoding(token_embeddings.size(2), max_len=token_embeddings.size(1))
        embeddings = PE(token_embeddings)
        # Combine token and position embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, vocab_size, embed_dim, num_heads=8, middle_dim=2048, drop_prob=0.1):
        super().__init__()
        self.embeddings = Embeddings(vocab_size, embed_dim, drop_prob)
        self.layers = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, middle_dim, drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, mask=None):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
