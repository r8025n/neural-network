from network import Network

def main():
    nn = Network([2, 3, 1])
    print(nn.weights)

if __name__ == "__main__":
    main()