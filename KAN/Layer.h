#pragma once
#include <memory>
#include <vector>
#include "Urysohn.h"

class Layer {
public:
	Layer(int nUrysohns, std::vector<double> xmin, std::vector<double> xmax, std::vector<double> tmin, std::vector<double> tmax, int nPoints) {
		if (xmin.size() != xmax.size()) {
			printf("Fatal: sizes of xmin, xmax mismatch\n");
			exit(0);
		}
		if (nUrysohns != tmin.size() || nUrysohns != tmax.size()) {
			printf("Fatal: sizes of tmin, tmax mismatch");
			exit(0);
		}
		_nFunctions_in_U = (int)xmin.size();
		for (int i = 0; i < nUrysohns; ++i) {
			_urysohns.push_back(std::make_unique<Urysohn>(xmin, xmax, tmin[i], tmax[i], nPoints));
		}
	}
	void Input2Output(const std::unique_ptr<double[]>& input, std::unique_ptr<double[]>& output, bool noUpdates = false) {
		for (int i = 0; i < _urysohns.size(); ++i) {
			output[i] = _urysohns[i]->GetValueUsingInput(input, noUpdates);
		}
	}
	void UpdateLayerUsingMemory(const std::unique_ptr<double[]>& deltas, double mu) {
		for (int i = 0; i < _urysohns.size(); ++i) {
			_urysohns[i]->UpdateUsingMemory(deltas[i] * mu);
		}
	}
	void GetDeltas(const std::unique_ptr<double[]>& deltas, std::unique_ptr<double[]>& derivatives) {
		for (int i = 0; i < _nFunctions_in_U; ++i) {
			derivatives[i] = 0.0;
		}
		for (int i = 0; i < _urysohns.size(); ++i) {
			_urysohns[i]->UpdateDerivativeVector(deltas[i], derivatives);
		}
	}
private:
	std::vector<std::unique_ptr<Urysohn>> _urysohns;
	int _nFunctions_in_U;
};

