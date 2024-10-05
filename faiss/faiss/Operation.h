#ifndef OPERATION_INDEX
#define OPERATION_INDEX

namespace faiss {
    
// // regex macro
// #include <iostream>
// #include <regex>
// // #define CHECK_REGEX(string, regex) \
// //   (std::regex_match(string, std::regex(regex)))

// #define CHECK_REGEX(string, regex) \
//    (!string.empty() && std::isdigit(string[0]))

enum Operation {
    EQUAL = 0,
    OR = 1,
    REGEX = 2,
};

} //namesapce faiss

// /// all vector indices are this type
// using idx_t = int64_t;

#endif