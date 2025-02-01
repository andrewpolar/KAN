#include <memory>
#include <vector>

//Bilinear Kaczmarz Approximation
class BiFunction {
public:
	BiFunction(double Fmin, double Fmax, int points) {
		_xmin = 1000000.0;  //this is done on purpose
		_xmax = -1000000.0; //same here
		_ymin = 1000000.0;  //same here
		_ymax = -1000000.0; //same here
		_points = points;
		SetLimits();
		InitializeRandom(Fmin, Fmax);
	}
	BiFunction(const BiFunction& bi) {
		_points = bi._points;
		_xmin = bi._xmin;
		_xmax = bi._xmax;
		_ymin = bi._ymin;
		_ymax = bi._ymax;
		_deltax = bi._deltax;
		_deltay = bi._deltay;
		_X = bi._X;
		_Y = bi._Y;
		_iX = bi._iX;
		_iY = bi._iY;
		_F.clear();
		for (int i = 0; i < bi._F.size(); ++i) {
			std::vector<double> x(bi._F[i].size());
			for (int j = 0; j < bi._F[i].size(); ++j) {
				x[j] = bi._F[i][j];
			}
			_F.push_back(x);
		}
	}
	double GetFunctionUsingInput(double x, double y) {
		FitDefinition(x, y);
		double offsetFromLeft = (x - _xmin) / _deltax;
		double offsetFromBottom = (y - _ymin) / _deltay;
		_iX = (int)(offsetFromLeft);
		_iY = (int)(offsetFromBottom);
		_X = offsetFromLeft - _iX;
		_Y = offsetFromBottom - _iY;
		double NW = _F[_iX][_iY + 1];
		double NE = _F[_iX + 1][_iY + 1];
		double SW = _F[_iX][_iY];
		double SE = _F[_iX + 1][_iY];
		return SW + _X * _Y * (SW - SE - NW + NE) + _X * (SE - SW) + _Y * (NW - SW);
	}
	void UpdateUsingMemory(double DF) {
		double wNW = (1.0 - _X) * _Y;
		double wNE = _X * _Y;
		double wSW = (1.0 - _X) * (1.0 - _Y);
		double wSE = _X * (1.0 - _Y);
		_F[_iX][_iY + 1] += DF * wNW;
		_F[_iX + 1][_iY + 1] += DF * wNE;
		_F[_iX][_iY] += DF * wSW;
		_F[_iX + 1][_iY] += DF * wSE;
	}
	void IncrementPoints() {
		std::vector<std::vector<double>> TMP(_points + 1, std::vector<double>(_points + 1, 0.0));
		for (int i = 0; i < _points; ++i) {
			for (int j = 0; j < _points; ++j) {
				TMP[i][j] = _F[i][j];
			}
		}
		for (int i = 0; i < _points + 1; ++i) {
			TMP[_points][i] = TMP[_points - 1][i];
		}
		for (int i = 0; i < _points + 1; ++i) {
			TMP[i][_points] = TMP[i][_points - 1];
		}
		_F.clear();
		for (int i = 0; i < TMP.size(); ++i) {
			std::vector<double> x(TMP[i].size());
			for (int j = 0; j < TMP[i].size(); ++j) {
				x[j] = TMP[i][j];
			}
			_F.push_back(x);
		}
		_points += 1;
		_deltax = (_xmax - _xmin) / (_points - 1);
		_deltay = (_ymax - _ymin) / (_points - 1);
		TMP.clear();
	}
	double GetDerivativeX() {
		return (_F[_iX + 1][_iY] - _F[_iX][_iY]) / _deltax;
	}
	double GetDerivativeY() {
		return (_F[_iX][_iY + 1] - _F[_iX][_iY]) / _deltay;
	}
private:
	double _xmin, _xmax, _ymin, _ymax, _deltax, _deltay, _X, _Y;
	int _points, _iX, _iY;
	std::vector<std::vector<double>> _F;
	void SetLimits() {
		double range = _xmax - _xmin;
		_xmin -= 0.01 * range;
		_xmax += 0.01 * range;
		range = _ymax - _ymin;
		_ymin -= 0.01 * range;
		_ymax += 0.01 * range;
		_deltax = (_xmax - _xmin) / (_points - 1);
		_deltay = (_ymax - _ymin) / (_points - 1);
	}
	void InitializeRandom(double Fmin, double Fmax) {
		_F.clear();
		for (int i = 0; i < _points; ++i) {
			std::vector<double> x;
			for (int j = 0; j < _points; ++j) {
				x.push_back(rand() % 100 / 100.0 * (Fmax - Fmin) + Fmin);
			}
			_F.push_back(x);
		}
	}
	void FitDefinition(double x, double y) {
		if (x < _xmin) {
			_xmin = x;
			SetLimits();
		}
		if (x > _xmax) {
			_xmax = x;
			SetLimits();
		}
		if (y < _ymin) {
			_ymin = y;
			SetLimits();
		}
		if (y > _ymax) {
			_ymax = y;
			SetLimits();
		}
	}
};
