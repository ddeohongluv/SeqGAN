from model import *

ACTIONS = ['[PAD]', '[SOS]', 'Logon', 'Logoff', 'Connect', 'Disconnect']
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
FILTER_SIZES = [2, 2]
NUM_FILTERS = [3, 3]
DROPOUT_RATE = .25
LEARNING_RATE = .01
DEVICE = torch.device('cuda')

if __name__ == '__main__': 

    vocab = Vocabulary(actions=ACTIONS)
    generator = Generator(vocab=vocab, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    discriminator = Discriminator(vocab=vocab, embedding_dim=EMBEDDING_DIM, filter_sizes=FILTER_SIZES, num_filters=NUM_FILTERS, dropout_rate=DROPOUT_RATE)
    
    # Pre-train generator
    gen_criterion = nn.NLLLoss(ignore_index=vocab.action_dict['[PAD]'])
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    gen_dataloader = something
    pre_train_generator(generator, gen_dataloader, gen_criterion, gen_optimizer, device)

    # Pre-train discriminator
    dis_criterion = nn.NLLLoss()
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    dis_dataloader = something
    pre_train_discriminator(discriminator, dis_dataloader, dis_criterion, dis_optimizer, device)

    # Adversarial training
    gen_criterion = PGLoss()
    dis_criterion = nn.NLLLoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    dis_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    for _ in range(TOTAL_REPEATS):
        policy = generator
        for _ in range(g_steps):
            source = torch.ones(size=(batch_size, 1)).long().to(DEVICE)
            generated_samples = generator.generate(source=source, target_len=MAX_LENGTH)
            rewards = action_value_function(generated_samples, policy, discriminator, MONTE_CARLO_NUM)
            log_probs = generator(generated_samples)
            log_probs = log_probs[:, :-1].gather(dim=-1, index=generated_samples[:, 1:].unsqueeze(-1)).squeeze()
            pg_loss = gen_criterion(log_probs, rewards)
            gen_optimizer.zero_grad()
            pg_loss.backward()
            gen_optimizer.step()

        for _ in range(d_steps):
            source = torch.ones(size=(batch_size, 1)).long().to(DEVICE)
            generated_samples = generator.generate(source=source, target_len=MAX_LENGTH)
            dataloader = something(generated_samples, real_samples)
            train_discriminator()

    save(generator)
