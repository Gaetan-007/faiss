#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <algorithm>

#include <faiss/MetricType.h>

namespace faiss {

struct HotListCache {
        std::unordered_map<int32_t, int32_t> mapping_table;
        std::unordered_map<int32_t, bool> activated; // to check if the listno is activated
        std::unordered_map<int32_t, int32_t> hits;
        std::vector<int32_t> listnos;
        size_t n;
        size_t activated_count = 0;
        HotListCache() : mapping_table(), activated(), n(0) {}

        void add_listno(idx_t listno) {
                int32_t listno32 = static_cast<int32_t>(listno);
                if (mapping_table.find(listno32) == mapping_table.end()) {
                        mapping_table[listno32] = mapping_table.size();
                        listnos.push_back(listno32);
                        activated[mapping_table.size() - 1] = true;
                        hits[mapping_table.size() - 1] = 0; // initialize hit count
                        activated_count++;
                        n++;
                }
        }

        void evict_listno(idx_t listno) {
                int32_t listno32 = static_cast<int32_t>(listno);
                if (n == 0) return;
                auto iter = mapping_table.find(listno32);
                if (iter != mapping_table.end()) {
                        listnos[iter->second] = -1;     // To Do fix it this is not working : Evicted listno is not removed from listnos
                        mapping_table.erase(listno32);
                        activated.erase(iter->second);
                        n--;
                }
        }

        void initialize_cache(std::string cache_file) {
                if (cache_file == "") {
                    n = 0;
                    return;
                }
                // load mapping table
                std::ifstream file(cache_file, std::ios::binary);
                if (!file) {
                        printf("Failed to open file for reading: %s\n", (cache_file).c_str());
                        return;
                }
                int32_t num;
                file.read(reinterpret_cast<char*>(&num), sizeof(int32_t));
                for (int32_t i = 0; i < num; i++) {
                        int32_t key, value;
                        file.read(reinterpret_cast<char*>(&key), sizeof(int32_t));
                        file.read(reinterpret_cast<char*>(&value), sizeof(int32_t));
                        listnos.push_back(key);
                        hits[value] = 0; // initialize hit count
                        mapping_table[key] = value;
                        activated[value] = true;
                        activated_count++;
                        n++;
                }
        }

        bool probe_cache(idx_t listno) {
                int32_t listno32 = static_cast<int32_t>(listno);
                if (mapping_table.find(listno32) == mapping_table.end()) {
                    return false; // not in cache
                } else if (activated.find(mapping_table.at(listno32)) == activated.end()) {
                    hits[mapping_table.at(listno32)]++; // increment hit count
                    return false; // not activated
                } else {
                    hits[mapping_table.at(listno32)]++; // increment hit count
                    return true; // in cache and activated
                }
        }

        void clear_cache() {
                mapping_table.clear();
                activated.clear();
                listnos.clear();
                hits.clear();
                activated_count = 0;
                n = 0;
        }

        void activate_listno(idx_t listno) {
                int32_t listno32 = static_cast<int32_t>(listno);
                if (mapping_table.find(listno32) != mapping_table.end()) {
                    activated[mapping_table.at(listno32)] = true;
                    activated_count++;
                }
        };

        void deactivate_listno(idx_t listno) {
                int32_t listno32 = static_cast<int32_t>(listno);
                if (mapping_table.find(listno32) != mapping_table.end()) {
                    activated[mapping_table.at(listno32)] = false;
                    activated_count--;
                }
        }

        // activate hottest n lists in the deactived lists
        void activate_n_lists(size_t n) {
                std::vector<std::pair<int32_t, int32_t>> hit_list(hits.begin(), hits.end());
                std::sort(hit_list.begin(), hit_list.end(), [](const auto& a, const auto& b) {
                        return a.second > b.second;
                });

                size_t actually_activated = 0;
                for (size_t i = 0; i < n && i < hit_list.size(); i++) {
                        // activate the listno if it is not already activated 
                        if (activated.find(hit_list[i].first) == activated.end() || !activated[hit_list[i].first]) {
                            activated[hit_list[i].first] = true;
                            actually_activated++;
                        }
                }
                activated_count += actually_activated;
                if (actually_activated < n) {
                        printf("Warning: Only %zu lists were activated, less than requested %zu.\n", actually_activated, n);
                }
        }

        // deactivate coldest n lists in the activated lists
        void deactivate_n_lists(size_t n) {
                std::vector<std::pair<int32_t, int32_t>> hit_list(hits.begin(), hits.end());
                std::sort(hit_list.begin(), hit_list.end(), [](const auto& a, const auto& b) {
                        return a.second < b.second;
                });

                size_t actually_deactivated = 0;
                for (size_t i = 0; i < n && i < hit_list.size(); i++) {
                        // deactivate the listno if it is activated
                        if (activated.find(hit_list[i].first) != activated.end() && activated[hit_list[i].first]) {
                            activated[hit_list[i].first] = false;
                            actually_deactivated++;
                        }
                }
                activated_count -= actually_deactivated;
                if (actually_deactivated < n) {
                        printf("Warning: Only %zu lists were deactivated, less than requested %zu.\n", actually_deactivated, n);
                }
        }
};
}