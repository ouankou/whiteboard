import sys

if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    print("Please provide a valid filename to process.")
    exit()

with open(filename) as f:
    startTime = int(next(f).split()[-1]) # read first line
    time = []
    for i in range(15):
        iterStart = int(next(f).split()[-1])
        iterEnd = int(next(f).split()[-1])
        time.append(iterEnd - iterStart)
    #print("Total time is: " + str(iterEnd - startTime))
    #print("Parsing time is: " + str(sum(time)))
    #print(time)
    print(str(iterEnd - startTime) + "," + str(sum(time)))
