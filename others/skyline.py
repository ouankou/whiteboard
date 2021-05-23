import heapq

class Hotel:
    def __init__(self, name, dis, price):
        self.name = name
        self.dis = dis
        self.price = price

def getVal(h):
    return h.dis + h.price

def isSkylinePoint(S, h):
    for v in S:
        if v.dis <= h.dis:
            if v.price <= h.price:
                if v.dis == h.dis and v.price == h.price:
                    return True
                else:
                    return False

    return True

if __name__ == '__main__':

    hotels = []
    hotels.append(Hotel("a", 1, 9))
    hotels.append(Hotel("b", 2, 10))
    hotels.append(Hotel("b", 2, 10));
    hotels.append(Hotel("c", 4, 8));
    hotels.append(Hotel("d", 6, 7));
    hotels.append(Hotel("e", 9, 10));
    hotels.append(Hotel("f", 7, 5));
    hotels.append(Hotel("g", 5, 6));
    hotels.append(Hotel("h", 4, 3));
    hotels.append(Hotel("i", 3, 2));
    hotels.append(Hotel("k", 9, 1));
    hotels.append(Hotel("l", 10, 4));
    hotels.append(Hotel("m", 6, 2));
    hotels.append(Hotel("n", 8, 3));
    hotels.sort(key=getVal, reverse=True)

    S = []
    while (hotels):
        tmp = hotels.pop()
        if isSkylinePoint(S, tmp):
            S.append(tmp)

    for v in S:
        print(v.name)
    
