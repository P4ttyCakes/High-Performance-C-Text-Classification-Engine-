#include <iostream>
using namespace std;
#include <set>
#include <fstream>
#include <sstream>
#include <cstring>
#include <iostream>
#include <cassert>
#include <array>
#include <string>
#include <vector>
#include <assert.h>
#include <algorithm>
#include  "csvstream.hpp"
#include <cmath>
#include <cctype>

// void BOW(int& n_o_p,map<string,int>&n_o_p_containing_word,
//     map<string,int>&num_of_content_with_tag, set<string>&total_words,
//     map<pair<string,string>,int> &pair_of_cw_count){

// }
// EFFECTS: Return a set of unique whitespace delimited words
set<string> unique_words(const string &str) {
  istringstream source(str);
  set<string> words;
  string word;
  while (source >> word) {
    words.insert(word);
  }
  return words;
}

// EFFECTS: Tokenize into whitespace-delimited words (lowercased).
static vector<string> tokenize_lower(const string &str) {
  istringstream source(str);
  vector<string> words;
  string word;
  while (source >> word) {
    for (char &c : word) c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
    words.push_back(word);
  }
  return words;
}
class Classifier{
    private:
        int n_o_p = 0; //number of posts
        map<string, int> n_o_p_containing_word;
        map<string,int> num_of_content_with_tag;
        set<string>total_words;
        map<pair<string,string>,int> pair_of_cw_count;

        // Improved model (Multinomial Naive Bayes with Laplace smoothing)
        map<pair<string, string>, int> word_count;
        map<string, int> total_words_in_class;
    public:
        void train(const string &filename, bool trainOnly){
            csvstream csvin(filename);
            map<string, string> row;

            while (csvin >> row) {
                n_o_p++;
                if (trainOnly){
                    cout << "  label = " << row["tag"]
                    << ", content = " << row["content"] << endl;
                }
                set<string>uw_set = unique_words(row["content"]);
                for (string s:uw_set) {
                    n_o_p_containing_word[s]+=1;
                    pair_of_cw_count[{row["tag"],s}]++;
                }
                num_of_content_with_tag[row["tag"]]++;
                //BOW[row["tag"]].insert(uw_set.begin(),uw_set.end());
                total_words.insert(uw_set.begin(),uw_set.end());
            }
        }

        // Improved training: uses token counts (not just presence/absence).
        void train_improved(const string &filename){
            csvstream csvin(filename);
            map<string, string> row;

            while (csvin >> row) {
                n_o_p++;
                const string &tag = row["tag"];
                num_of_content_with_tag[tag]++;

                vector<string> toks = tokenize_lower(row["content"]);
                for (const string &w : toks) {
                    word_count[{tag, w}]++;
                    total_words_in_class[tag]++;
                    total_words.insert(w);
                }
            }
        }
        void print_results(bool trainOnly){
            cout << "trained on " << n_o_p << " examples" << endl;
            if (trainOnly){
                cout << "vocabulary size = " << total_words.size() <<endl<<endl;
                cout << "classes:"<<endl;
                for (const auto &A: num_of_content_with_tag){
                string tag = A.first;
                int count = A.second;
                double log_temp = log((double)count / n_o_p);
                cout << "  " << tag << ", " << count
                     << " examples, log-prior = " << log_temp << endl;

                }
                cout << "classifier parameters:"<<endl;
                for (const auto &A: pair_of_cw_count){
                string tag = A.first.first;
                string word = A.first.second;
                int count = A.second;
                double log_temp = log((double)count / num_of_content_with_tag.at(tag));
                cout << "  " << tag<< ":" << word
                     << ", count = " << count
                     << ", log-likelihood = " << log_temp << endl;
            }
             cout << endl;
            }

        }

    double find_Score(string tag, const set<string> &words){
        double score = log((double)num_of_content_with_tag.at(tag) / n_o_p);
        for (const string &A : words) {
            int posts_w_word = 0;
            if (n_o_p_containing_word.count(A) > 0){
            posts_w_word = n_o_p_containing_word.at(A);
            }
            int count = 0;
        if (pair_of_cw_count.count({tag, A}) > 0) count = pair_of_cw_count.at({tag, A});
            double log_count;
            if (posts_w_word == 0) {
                log_count = log(1.0 / n_o_p);
            }
            else if (count == 0) {
                log_count = log((double)posts_w_word / n_o_p);
            }
            else {
                log_count = log((double)count /
                                num_of_content_with_tag.at(tag));
            }
            score += log_count;
        }
        return score;
    }

    // Improved scoring: Multinomial Naive Bayes with Laplace smoothing.
    double find_Score_improved(const string &tag, const vector<string> &tokens) const {
        const int V = static_cast<int>(total_words.size());
        double score = log((double)num_of_content_with_tag.at(tag) / n_o_p);

        const int denom = total_words_in_class.at(tag) + V;
        for (const string &w : tokens) {
            int c = 0;
            auto it = word_count.find({tag, w});
            if (it != word_count.end()) c = it->second;
            score += log((double)(c + 1) / denom);
        }
        return score;
    }

        string predict(const string &content){
            set<string> words = unique_words(content);
            string label;
            bool first = true;
            double score = -INFINITY;
            for (const auto& A : num_of_content_with_tag){
                const string &tag = A.first;
                double temp = find_Score(tag, words);
                if (first){
                    score = temp;
                    label = tag;
                    first = false;
                }
                if (temp>score) {
                    score = temp;
                    label = tag;
                } else if (score == temp && tag < label){
                    label = tag;
                }
            }
            return label;
        }

        string predict_improved(const string &content) const {
            vector<string> tokens = tokenize_lower(content);
            string label;
            bool first = true;
            double best = -INFINITY;
            for (const auto& A : num_of_content_with_tag){
                const string &tag = A.first;
                double s = find_Score_improved(tag, tokens);
                if (first || s > best || (s == best && tag < label)) {
                    best = s;
                    label = tag;
                    first = false;
                }
            }
            return label;
        }
};
int main(int argc, char* argv[]) {
    cout.precision(3);
    bool improved = false;
    if (argc >= 2 && strcmp(argv[argc - 1], "--improved") == 0) {
        improved = true;
        argc -= 1; // ignore flag for positional parsing
    }
    if (argc!=2 && argc!=3) {
        cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE] [--improved]" << endl;
        return 1;
    }
    ifstream input; ifstream input2;
    input.open(argv[1]);
    if (argc==3) input2.open(argv[2]);
    if(!input.is_open() || (!input2.is_open() && argc==3)){
        cout <<"Error opening file: "<< argv[1] << endl;
        return 1;
    }
    string filename = argv[1];
    Classifier classifier;
    bool testOnly = argc==2;
    if (testOnly) cout << "training data:" <<endl;
    if (improved) {
        classifier.train_improved(filename);
    } else {
        classifier.train(filename, testOnly);
    }
    classifier.print_results(testOnly);
    if (argc==3) {
    string test_file = argv[2];
        cout << endl;
        cout << "test data:" << endl;
        csvstream csvin(test_file);
        map<string, string> row;
        int correct = 0;
        int total = 0;
        while (csvin >> row) {
            string clabel = row["tag"];
            string content = row["content"];
            string predicted;
            double score;
            if (improved) {
                predicted = classifier.predict_improved(content);
                score = classifier.find_Score_improved(predicted, tokenize_lower(content));
            } else {
                predicted = classifier.predict(content);
                score = classifier.find_Score(predicted, unique_words(content));
            }
            cout << "  correct = " << clabel
                 << ", predicted = " << predicted
                 << ", log-probability score = " << score << endl;
            cout << "  content = " << content << endl<<endl;
            total++;
                if (predicted == clabel) correct++;
            }
            cout << "performance: " << correct << " / " <<total
                 << " posts predicted correctly" <<endl;
        }
        return 0;
}
