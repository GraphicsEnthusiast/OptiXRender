#include "SceneParser.h"

void SceneParser::LoadFromJson(const string& fn, Scene& scene, bool& is_succeed) {
	json data = CreateJsonFromFile(fn, is_succeed);
	if (!is_succeed) {
		return;
	}
	Parse(data, scene);
}

void SceneParser::Parse(const json& data, Scene& scene) {
	width = data.value("width", 1024);
	height = data.value("height", 1024);
	depth = data.value("depth", 4);

	cout << "Settings Information: " << endl;
	cout << "width: " << width << endl;
	cout << "height: " << height << endl;
	cout << "depth: " << depth << endl;

 	json filterData = data.value("filter", json());
 	ParseFilter(filterData);
}

void SceneParser::ParseFilter(const json& data) {
	cout << "Filter Information: " << endl;
	string filterType = data.value("type", "gaussian");

	cout << "type: " << filterType << endl << endl;

	if (filterType == "gaussian") {
		filterType = FilterType::Gaussian;
	}
	else if (filterType == "triangle") {
		filterType = FilterType::Triangle;
	}
	else if (filterType == "tent") {
		filterType = FilterType::Tent;
	}
	else if (filterType == "box") {
		filterType = FilterType::Box;
	}
	else {
		filterType = FilterType::Gaussian;
	}
}
