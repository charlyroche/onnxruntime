import argparse
import torch
from torchvision import datasets, transforms

import onnxruntime
from onnxruntime.training import ORTModule

import _test_commons
import _test_helpers


class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input1):
        # print(f'{"*"*128}\n RED ALERT! MNIST FORWARD METHOD WAS CALLED\n{"*"*128}')
        out = self.fc1(input1)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 20)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--pytorch-only', action='store_true', default=False,
                        help='disables ONNX Runtime training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status (default: 100)')
    parser.add_argument('--view-graphs', action='store_true', default=False,
                        help='views forward and backward graphs')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()

    # Model architecture
    torch.manual_seed(args.seed)
    onnxruntime.set_seed(args.seed)

    model = NeuralNet(input_size=784, hidden_size=500, num_classes=10)
    print('Training MNIST on ORTModule....')
    if not args.pytorch_only:
        model = ORTModule(model)

    criterion = torch.nn.CrossEntropyLoss()
    print(f'Model parameters: {[p[0] for p in model.named_parameters()]}')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Data loader
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
                                            transform=transforms.Compose([transforms.ToTensor(),

                                                                            transforms.Normalize((0.1307,), (0.3081,))])),
                                            batch_size=args.batch_size,
                                            shuffle=True)
    # Training Loop
    loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        for iteration, (data, target) in enumerate(train_loader):
            data = data.reshape(data.shape[0], -1)
            optimizer.zero_grad()
            if args.pytorch_only:
                # print("Using PyTorch-only API")
                probability = model(data)
            else:
                # print("Using ONNX Runtime Flexible API")
                probability = model(data)

            if args.view_graphs:
                import torchviz
                pytorch_backward_graph = torchviz.make_dot(probability, params=dict(list(model.named_parameters())))
                pytorch_backward_graph.view()

            # print(f'Output from forward (probability) has shape {probability.size()} and requires_grad={probability.requires_grad}')
            probability.retain_grad() # We want to print it later just for fun

            loss = criterion(probability, target)
            loss.backward()
            # print(f'probability\n\t grad_fn={probability.grad_fn}\n\t grad_fn.next_functions={probability.grad_fn.next_functions}\n\t probability.grad[0]={probability.grad[0]}')
            optimizer.step()

            # Stats
            if iteration % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, iteration * len(data), len(train_loader.dataset),
                    100. * iteration / len(train_loader), loss))

    print(f'Tah dah! Final loss={loss}')

if __name__ == '__main__':
    main()
