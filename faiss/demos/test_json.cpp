#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>      // for ostringstream
#include <fstream> 

using json = nlohmann::json;
int main(){
    std::string filepath = "/home/mocheng/code/test/testjson.json";
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open JSON file" << std::endl;
        // return 1;
    }

    json data;
    try {
        file >> data;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse JSON data from " << filepath << ": " << e.what() << std::endl;
        // return 1;
    }

    std::vector<int> v =  data.get<std::vector<int>>();

    std::cout << "get val ";
    for(auto i : v){
        std::cout << i << " ";
    }
    std::cout << std::endl;
}