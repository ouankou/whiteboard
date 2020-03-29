import sys
import json
import csv

if __name__ == "__main__":

    if len(sys.argv) == 3:
        csvFilename = sys.argv[1]
        jsonFilename = sys.argv[2]
    else:
        print("Please provide a valid filename to process.")
        exit()
    
    csvFile = open(csvFilename, 'r')
    jsonFile = open(jsonFilename, 'w')

    header = next(csvFile)
    if header[-1] == '\n':
        header = header[:-1] 
    header = header.split(',')
    print(header)
    result = {}
    report = {'tsan':result}
    index = 0
    length = len(header)
    row1 = next(csvFile, None)
    row2 = next(csvFile, None)

    while row1 and row2:
#        if index > 5:
#            break
        row = row1[:-1] + ',' + row2[:-1]
        data = row.split(',')
        #print(data)
        row_result = {}
        for i in range(length):
            row_result[header[i]] = data[i]
        result[str(index)] = row_result
        index += 1
        row1 = next(csvFile, None)
        row2 = next(csvFile, None)
    print(report)

    jsonFile.write(json.dumps(report, indent=4))

