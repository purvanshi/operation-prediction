import argparse


def readLinesFromFile(filePath):
    with open(filePath, 'r') as fileRead:
        return [line.strip() for line in fileRead.readlines() if line.strip()]


def processLinesAndFindAccuracy(lines):
    return sum([1 for line in lines if line.split(',')[0].strip() == line.split(',')[1].strip()]) / len(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='inp', help='Enter the predictions and gold file path')
    args = parser.parse_args()
    print(processLinesAndFindAccuracy(readLinesFromFile(args.inp)))
