import torch

if __name__ == '__main__':
    inp = torch.load("models/input.pt")

    model = torch.jit.load("models/u2net.pt", map_location="cpu")
