import torch
from torch import utils, nn
import random


def syntheticData(weight, bias, label_length):
    X: torch.Tensor = torch.normal(0, 1, (label_length, len(weight)))
    y: torch.Tensor = torch.matmul(X, weight) + bias
    # Generate noise
    y += torch.normal(0, 0.01, y.shape)

    return X, y.reshape((-1, 1))


def main():
    weight = torch.tensor([2, -3.4])
    bias = 4.2
    features, labels = syntheticData(weight, bias, 1000)

    # Read data
    batch_size = 10
    dataset = utils.data.TensorDataset(*(features, labels))
    data_iter = utils.data.DataLoader(dataset, batch_size, shuffle=True)

    # Construct network
    net = nn.Sequential(nn.Linear(2, 1))
    # Init network
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    # Loss
    loss = nn.MSELoss()

    # Algorithm
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    # Training
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss {l:f}")

    # Output
    w = net[0].weight.data
    print("w的估计误差：", weight - w.reshape(weight.shape))
    b = net[0].bias.data
    print("b的估计误差：", bias - b)


if __name__ == "__main__":
    main()
