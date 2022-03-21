from matplotlib import pyplot as plt

train_losses = range(1, 11)
train_bleu_scores = range(1, 11)
fig = plt.figure(figsize=(24, 8))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.plot(list(range(len(train_losses))), train_losses, label="train loss")
ax2.plot(
    list(range(len(train_bleu_scores))),
    train_bleu_scores,
    label="train bleu scores",
)
ax1.legend()
ax2.legend()
plt.savefig("train_loss_and_bleu_scores.png")
