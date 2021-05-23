#include<iostream>
#include<vector>
#include<queue>
using namespace std;

// Usage: clang++ skyline.cpp -o skyline.out
// ./skyline.out

/*
*   定义一个hotel
*/
struct Hotel{
    string name;
    int dis;
    int price;

    Hotel(string s, int d, int p){
        name = s;
        dis = d;
        price = p;
    }
};

struct cmp{
    bool operator () (const Hotel &a, const Hotel &b){
        return a.dis + a.price > b.dis + b.price;
    }
};

bool isSkyLinePoint(vector<Hotel> S, Hotel h){
    for(int i = 0; i < S.size(); i++){
        if(S[i].dis <= h.dis){
            if(S[i].price <= h.price){
                if(S[i].price == h.price && S[i].dis == h.dis) return 1;
                return 0;
            }
        }
    }

    return 1;
}

int main(){

    //存放所有hotel的优先队列
    priority_queue<Hotel, vector<Hotel>, cmp> hotels;

    hotels.push(Hotel("a", 1, 9));
    hotels.push(Hotel("b", 2, 10));
    hotels.push(Hotel("c", 4, 8));

    hotels.push(Hotel("d", 6, 7));
    hotels.push(Hotel("e", 9, 10));
    hotels.push(Hotel("f", 7, 5));

    hotels.push(Hotel("g", 5, 6));
    hotels.push(Hotel("h", 4, 3));
    hotels.push(Hotel("i", 3, 2));

    hotels.push(Hotel("k", 9, 1));
    hotels.push(Hotel("l", 10, 4));
    hotels.push(Hotel("m", 6, 2));

    hotels.push(Hotel("n", 8, 3));

    //依次从最小堆中取出数据，并判断S集中的point能不能dominate这个需要判断的点
    vector<Hotel> S;

    while(!hotels.empty()){
        Hotel tmp = hotels.top();
        hotels.pop();

        if(isSkyLinePoint(S, tmp))
            S.push_back(tmp);
    }

    for(int i = 0; i < S.size(); i++)
        cout<<S[i].name<<endl;

    return 0;
}
