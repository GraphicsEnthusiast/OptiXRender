#pragma once

#include "Scene.h"
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <nlohmann/json.hpp>

using namespace std;
using namespace nlohmann;

inline bool HasExtension(const string& value, const string& ending) {
	if (ending.size() > value.size()) {
		return false;
	}

	return std::equal(
		ending.rbegin(), ending.rend(), value.rbegin(),
		[](char a, char b) { return std::tolower(a) == std::tolower(b); });
}

inline json CreateJsonFromFile(const string& fn, bool& is_succeed) {
	std::ifstream fst;
	fst.open(fn.c_str());
	is_succeed = false;
	fst.exceptions(ifstream::failbit || ifstream::badbit);
	if (!fst.is_open()) {
		printf("Open scene json failed! \n");

		return json();
	}
	else {
		is_succeed = true;
		std::stringstream buffer;
		buffer << fst.rdbuf();
		std::string str = buffer.str();
		if (HasExtension(fn, "bson")) {
			return json::from_bson(str);
		}
		else {
			return json::parse(str);
		}
	}
}

class SceneParser {
public:
	SceneParser() = default;

	void LoadFromJson(const string& fn, Scene& scene, bool& is_succeed);
	void Parse(const json& data, Scene& scene);
	void ParseFilter(const json& data);

public:
	int width, height, depth;
	FilterType filterType;
};