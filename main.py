from network import Network

def main():
    nn = Network([2, 3, 1])
    print(nn.weights)
    nn.feed_forward([1, 2])

if __name__ == "__main__":
    main()
    