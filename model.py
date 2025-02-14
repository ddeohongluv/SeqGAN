import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Vocabulary(object):
    """
    container of action-ids
    """
    def __init__(self, actions: list):
        """
        Args:
            actions: list of action-ids (including special tokens e.g. [SOS], [PAD])
        """
        self.actions = actions

        self.action_dict = self._generate_action_dict()

    def __len__(self):

        return len(self.actions)

    def _generate_action_dict(self):
        actions = self.actions

        return {action_id:idx for idx, action_id in enumerate(actions)}

class SequenceEmbedding(nn.Module):
    """
    sequence embedding
    """
    def __init__(self, vocab: Vocabulary, embedding_dim: int = None):
        """
        Args:
            vocab: `Vocabulary` instance
            embedding_dim: embedding 벡터 차원
        """
        super().__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim

        self.pad_id = vocab.action_dict['[PAD]']
        self.sos_id = vocab.action_dict['[SOS]']
        self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=self.pad_id)
   
    def forward(self, x):
        """
        x: (batch_size, max_len)
        """
        embedded = self.embedding(x) # (batch_size, max_len) -> (batch_size, max_len, embedding_dim)

        return embedded

class Generator(nn.Module):
    """
    생성기
    """
    def __init__(self, vocab: Vocabulary, embedding_dim, hidden_dim):
        """
        Args:
            vocab: `Vocabulary` instance
            embedding_dim: embedding 벡터 차원
            hidden_dim: LSTM 은닉 벡터 차원
        """
        super().__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = SequenceEmbedding(vocab, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, len(vocab))
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: Tensor):
        """
        파라미터를 학습하기 위한 순전파 과정
        합성 데이터를 생성하기 위해, 실제 데이터의 패턴을 LSTM이 학습
        Args:
            x: (batch_size, max_len), 실제 데이터 (start-of-sentence token 포함)
        """
        embedded = self.embedding(x)

        lengths = x.ne(self.embedding.pad_id).sum(axis=1).cpu()
        packed_embedded = pack_padded_sequence(embedded, lengths=lengths, batch_first=True, enforce_sorted=False)

        packed_output, _ = self.lstm(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True) # (batch_size, max_len, hidden_dim)
        
        log_probs = self.log_softmax(self.linear(output)) # NLLLoss 사용을 위한 log_softmax (batch_size, max_len, len(vocab))

        # criterion = NLLLoss(ignore_index=pad_id)
        # sequences = somthing with (batch_size, max_len)
        # target = torch.cat([sequences[:,1:], torch.full(size=(batch_size, 1), fill_value=pad_id, device=sequences.device)], dim=1)
        # log_probs = generator(sequences)
        # loss = criterion(log_probs, target)
        # loss.backward()

        return log_probs

    def _step(self, given_tokens: Tensor, h: Tensor = None, c: Tensor = None):
        """
        Args:
            given_tokens: (batch_size,  1), sequence of tokens given
            h: (1, batch_size, hidden_dim), LSTM current hidden state
            c: (1, batch_size, hidden_dim), LSTM current cell state
        """
        embedded = self.embedding(given_tokens)
        if h is None or c is None: 
            output, (h, c) = self.lstm(embedded) # (batch_size, 1, hidden_dim)
        else:
            output, (h, c) = self.lstm(embedded, (h, c)) # (batch_size, 1, hidden_dim)
        logits = self.linear(output.squeeze()) # (batch_size, len(vocab))
        probs = F.softmax(logits, dim=-1)
        new_tokens = probs.multinomial(1) # (batch_size, 1)

        return new_tokens, h, c

    def generate(self, source: Tensor, target_len: int):
        """
        Args:
            source: (batch_size, source_len), 시퀀스 생성을 위해 주어진 초기 값 (sos token 포함)
            target_len: 생성할 시퀀스의 길이
        """
        _, source_len = source.shape
        samples = []

        self.eval()
        with torch.no_grad():
            h, c = (None, None)
            for idx in range(source_len):
                samples.append(source[:, [idx]])
                output, h, c = self._step(source[:, [idx]], h, c)

            for idx in range(source_len, target_len):
                samples.append(output)
                if idx < target_len - 1:
                    output, h, c = self._step(output, h, c)

            samples = torch.cat(samples, dim=1)

            for idx, sample in enumerate(samples):
                pad_mask = sample[1:].eq(self.embedding.pad_id)
                sos_mask = sample[1:].eq(self.embedding.sos_id)
                pad_positions = (pad_mask | sos_mask).nonzero(as_tuple=False)
                if pad_positions.shape[0]:
                    sample[1:][pad_positions[0][0]:] = self.embedding.pad_id
            
            valid_positions = samples.ne(self.embedding.pad_id).sum(dim=0).ne(0) # to delete too much padding
            samples = samples[:, valid_positions]

        self.train()

        return samples

class Discriminator(nn.Module):
    """
    판별기
    구조: embedding >> convolution >> max-pooling >> softmax
    """
    def __init__(self, vocab: Vocabulary, embedding_dim, filter_sizes, num_filters, dropout_rate):
        """
        Args:
            vocab: `Vocabulary` instance
            embedding_dim: embedding 벡터 차원
            filter_sizes: 필터의 가로 길이 (shape of filter = (filter_size, embedding_dim))
            num_filters: 사용할 필터 개수
            hidden_dim: LSTM 은닉 벡터 차원
        """
        super().__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate

        self.embedding = SequenceEmbedding(vocab, embedding_dim)
        self.convolutions = nn.ModuleList([
            nn.Conv2d(1, num_filter, (filter_size, embedding_dim)) for (num_filter, filter_size) in zip(num_filters, filter_sizes)
        ])
        total_num_filters = sum(num_filters)
        self.highway = nn.Linear(total_num_filters, total_num_filters)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(total_num_filters, 2) # 'Real' or 'Fake'?
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: (batch_size, max_len)
        """
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1) # (batch_size, 1, max_len, embedding_dim)
        outputs = [F.relu(conv(embedded)).squeeze(-1) for conv in self.convolutions] # list of (batch_size, num_filter, length)
        outputs = [F.max_pool1d(output, output.shape[-1]).squeeze(-1) for output in outputs] # list of (batch_size, num_filter)
        output = torch.cat(outputs, dim=1) # (batch_size, total_num_filters)

        # highway layer
        highway = self.highway(output)
        output = F.sigmoid(highway) * F.relu(highway) + (1. - F.sigmoid(highway)) * output

        output = self.log_softmax(self.linear(self.dropout(output))) # (batch_size, 2)

        return output
    
class PGLoss(nn.Module):
    """
    Policy-Gradient Loss (Expected reward)
    """
    def __init__(self):
        super().__init__()

    def forward(self, log_probs, rewards):
        """
        Args:
            log_probs: (batch_size, max_len), Y_{1:t-1}이 주어졌을 때 각각의 y_t가 생성될 확률
            rewards: (batch_size, max_len), 각 Y_{1:t-1}과 y_t 에 대한 Q 값. padding 부분에 해당하는 reward는 0
        """

        loss = log_probs * rewards # (batch_size, max_len), padding 부분에 해당하는 reward는 0
        lengths = loss.ne(0).sum(dim=1).unsqueeze(dim=1) + 1e-9
        loss = loss / lengths # (batch_size, max_len)

        return -loss.sum(dim=1).mean()
    
def action_value_function(generated_samples: Tensor, policy: Generator, discriminator: Discriminator, monte_carlo_num: int):
    """
    Args:
        generated_samples: (batch_size, max_len), sos token 포함
        policy: roll-out policy (same as the generator)
        discriminator: the discriminator as a expected cumulative reward
    Output:
        rewards: (batch_size, max_len - 1), sos token 미포함함
    """
    _, max_len = generated_samples.shape
    pad_id = policy.embedding.pad_id
    pad_mask = generated_samples.eq(pad_id)
    rewards = []
    discriminator.eval()
    for t in range(2, max_len):
        source = generated_samples[:, :t]
        reward = 0
        for _ in range(monte_carlo_num):
            monte_carlo_samples = policy.generate(source, target_len=max_len) # (batch_size, max_len)
            with torch.no_grad():
                reward += discriminator(monte_carlo_samples)[:, 1].exp() # (batch_size,)
        reward /= monte_carlo_num
        rewards.append(reward)

    # When t = max_len
    with torch.no_grad():
        reward = discriminator(generated_samples)[:, 1].exp()
    rewards.append(reward)
    rewards = torch.stack(rewards, dim=1) # (batch_size, max_len - 1)
    rewards *= ~pad_mask[:, 1:]

    return rewards