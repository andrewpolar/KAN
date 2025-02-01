#pragma once
#include <memory>
#include "Function.h"
#include "Helper.h"

class Urysohn
{
public:
	Urysohn(std::vector<double> xmin, std::vector<double> xmax, double targetMin, double targetMax, int points_in_function) {
		if (xmin.size() != xmax.size()) {
			printf("Fatal: wrong array sizes\n");
			exit(0);
		}
		int size = (int)xmin.size();
		double oneFunctionMin = targetMin / size;
		double oneFunctionMax = targetMax / size;
		Helper::Sum2IndividualLimits(targetMin, targetMax, size, oneFunctionMin, oneFunctionMax);
		for (int i = 0; i < size; ++i) {
			_functionList.push_back(std::make_unique<Function>(xmin[i], xmax[i],
				oneFunctionMin, oneFunctionMax, points_in_function));
		}
	}
	Urysohn(const Urysohn& uri) {
		_functionList.clear();
		for (int i = 0; i < uri._functionList.size(); ++i) {
			_functionList.push_back(std::make_unique<Function>(*uri._functionList[i]));
		}
	}
	void UpdateUsingInput(double delta, const std::unique_ptr<double[]>& inputs) {
		for (int i = 0; i < _functionList.size(); ++i) {
			_functionList[i]->UpdateUsingInput(inputs[i], delta);
		}
	}
	void UpdateUsingMemory(double delta) {
		for (int i = 0; i < _functionList.size(); ++i) {
			_functionList[i]->UpdateUsingMemory(delta);
		}
	}
	double GetValueUsingInput(const std::unique_ptr<double[]>& inputs, bool noUpdates = false) {
		double f = 0.0;
		for (int i = 0; i < _functionList.size(); ++i) {
			f += _functionList[i]->GetFunctionUsingInput(inputs[i], noUpdates);
		}
		return f;
	}
	void IncrementInner() {
		for (int i = 0; i < _functionList.size(); ++i) {
			_functionList[i]->IncrementPoints();
		}
	}
	void UpdateDerivativeVector(const double delta, std::unique_ptr<double[]>& derivatives) {
		for (int i = 0; i < _functionList.size(); ++i) {
			derivatives[i] += delta * _functionList[i]->GetDerivativeUsingMemory();
		}
	}
private:
	std::vector<std::unique_ptr<Function>> _functionList;
};