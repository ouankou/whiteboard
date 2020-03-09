import sys

if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    print("Please provide a valid filename to process.")
    exit()

with open(filename) as f:
    startTime = int(next(f).split()[-1]) # read first line
    time = []
    outliers = 0
    for i in range(15):
        iterStart = int(next(f).split()[-1])
        iterEnd = int(next(f).split()[-1])
        if (iterEnd - iterStart) < 0:
            outliers += 1
        else:
            time.append(iterEnd - iterStart)
    if outliers:
        makeup = sum(time)/len(time)
        for i in range(outliers):
            time.append(makeup)
    
    #print("Total time is: " + str(iterEnd - startTime))
    #print("Parsing time is: " + str(sum(time)))
    #print(time)
    if iterEnd:
        print(str(iterEnd - startTime) + "," + str(sum(time)))
