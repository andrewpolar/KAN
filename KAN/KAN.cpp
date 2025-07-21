//Concept: Andrew Polar and Mike Poluektov
//Developer Andrew Polar

// License
// If the end user somehow manages to make billions of US dollars using this code,
// and happens to meet the developer begging for change outside a McDonald's,
// he or she is under no obligation to buy the developer a sandwich.

// Symmetry Clause
// Likewise, if the developer becomes rich and famous by publishing this code,
// and meets an unfortunate end user who went bankrupt using it,
// the developer is also under no obligation to buy the end user a sandwich..

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://arxiv.org/abs/2305.08194

#include <iostream>
#include <vector>
#include "Helper.h"
#include "Layer.h"

//determinants
std::unique_ptr<std::unique_ptr<double[]>[]> GenerateInput(int nRecords, int nFeatures, double min, double max) {
	auto x = std::make_unique<std::unique_ptr<double[]>[]>(nRecords);
	for (int i = 0; i < nRecords; ++i) {
		x[i] = std::make_unique<double[]>(nFeatures);
		for (int j = 0; j < nFeatures; ++j) {
			x[i][j] = static_cast<double>((rand() % 10000) / 10000.0);
			x[i][j] *= (max - min);
			x[i][j] += min;
		}
	}
	return x;
}

double determinant(const std::vector<std::vector<double>>& matrix) {
	int n = (int)matrix.size();
	if (n == 1) {
		return matrix[0][0];
	}
	if (n == 2) {
		return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
	}
	double det = 0.0;
	for (int col = 0; col < n; ++col) {
		std::vector<std::vector<double>> subMatrix(n - 1, std::vector<double>(n - 1));
		for (int i = 1; i < n; ++i) {
			int subCol = 0;
			for (int j = 0; j < n; ++j) {
				if (j == col) continue;
				subMatrix[i - 1][subCol++] = matrix[i][j];
			}
		}
		det += (col % 2 == 0 ? 1 : -1) * matrix[0][col] * determinant(subMatrix);
	}
	return det;
}

double ComputeDeterminant(std::unique_ptr<double[]>& input, int N) {
	std::vector<std::vector<double>> matrix(N, std::vector<double>(N, 0.0));
	int cnt = 0;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			matrix[i][j] = input[cnt++];
		}
	}
	return determinant(matrix);
}

std::unique_ptr<double[]> ComputeDeterminantTarget(const std::unique_ptr<std::unique_ptr<double[]>[]>& x, int nMatrixSize, int nRecords) {
	auto target = std::make_unique<double[]>(nRecords);
	int counter = 0;
	while (true) {
		target[counter] = ComputeDeterminant(x[counter], nMatrixSize);
		if (++counter >= nRecords) break;
	}
	return target;
}
//end

std::unique_ptr<std::unique_ptr<double[]>[]> MakeRandomMatrix(int rows, int cols, double min, double max) {
	std::unique_ptr<std::unique_ptr<double[]>[]> matrix;
	matrix = std::make_unique<std::unique_ptr<double[]>[]>(rows);
	for (int i = 0; i < rows; ++i) {
		matrix[i] = std::make_unique<double[]>(cols);
		for (int j = 0; j < cols; ++j) {
			matrix[i][j] = static_cast<double>((rand() % 1000) / 1000.0) * (max - min) + min;
		}
	}
	return matrix;
}
double AreaOfTriangle(double x1, double y1, double x2, double y2, double x3, double y3) {
	double A = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
	return A;
}
std::unique_ptr<double[]> ComputeAreas(std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, int N) {
	auto u = std::make_unique<double[]>(N);
	for (int i = 0; i < N; ++i) {
		u[i] = AreaOfTriangle(matrix[i][0], matrix[i][1], matrix[i][2], matrix[i][3], matrix[i][4], matrix[i][5]);
	}
	return u;
}

void Determinant_4_4_3Layers() {
	printf("Target is determinants of random 4*4 matrices, model is 3 layer KAN, metric is Pearson\n");
	//generate data
	int nTrainingRecords = 100000;
	int nValidationRecords = 20000;
	int nMatrixSize = 4;
	int nFeatures = nMatrixSize * nMatrixSize;
	double min = 0.0;
	double max = 10.0;
	auto inputs_training = GenerateInput(nTrainingRecords, nFeatures, min, max);
	auto inputs_validation = GenerateInput(nValidationRecords, nFeatures, min, max);
	auto target_training = ComputeDeterminantTarget(inputs_training, nMatrixSize, nTrainingRecords);
	auto target_validation = ComputeDeterminantTarget(inputs_validation, nMatrixSize, nValidationRecords);

	//data is ready, we start training
	clock_t start_application = clock();
	clock_t current_time = clock();

	std::vector<double> xmin;
	std::vector<double> xmax;
	Helper::FindMinMaxMatrix(xmin, xmax, inputs_training, nTrainingRecords, nFeatures);
	double targetMin = Helper::Min(target_training, nTrainingRecords);
	double targetMax = Helper::Max(target_training, nTrainingRecords);

	int nU0 = 128;
	int nU1 = 4;
	int nU2 = 1;

	std::vector<double> tmin0;
	std::vector<double> tmax0;
	for (int i = 0; i < nU0; ++i) {
		tmin0.push_back(targetMin);
		tmax0.push_back(targetMax);
	}

	std::vector<double> tmin1;
	std::vector<double> tmax1;
	for (int i = 0; i < nU1; ++i) {
		tmin1.push_back(targetMin);
		tmax1.push_back(targetMax);
	}

	std::vector<double> tmin2;
	std::vector<double> tmax2;
	for (int i = 0; i < nU2; ++i) {
		tmin2.push_back(targetMin);
		tmax2.push_back(targetMax);
	}

	auto layer0 = std::make_unique<Layer>(nU0, xmin, xmax, tmin0, tmax0, 3);
	auto layer1 = std::make_unique<Layer>(nU1, tmin0, tmax0, tmin1, tmax1, 6);
	auto layer2 = std::make_unique<Layer>(nU2, tmin1, tmax1, tmin2, tmax2, 12);

	auto models0 = std::make_unique<double[]>(nU0);
	auto models1 = std::make_unique<double[]>(nU1);
	auto models2 = std::make_unique<double[]>(nU2);
	auto deltas0 = std::make_unique<double[]>(nU0);
	auto deltas1 = std::make_unique<double[]>(nU1);
	auto deltas2 = std::make_unique<double[]>(nU2);

	auto model_training = std::make_unique<double[]>(nTrainingRecords);
	auto model_validation = std::make_unique<double[]>(nValidationRecords);
	for (int epoch = 0; epoch < 12; ++epoch) {
		for (int i = 0; i < nTrainingRecords; ++i) {
			layer0->Input2Output(inputs_training[i], models0);
			layer1->Input2Output(models0, models1);
			layer2->Input2Output(models1, models2);
			deltas2[0] = target_training[i] - models2[0];
			model_training[i] = models2[0];
			layer2->GetDeltas(deltas2, deltas1);
			layer1->GetDeltas(deltas1, deltas0);
			layer2->UpdateLayerUsingMemory(deltas2, 0.01);
			layer1->UpdateLayerUsingMemory(deltas1, 0.05);
			layer0->UpdateLayerUsingMemory(deltas0, 0.2);
		}
		if (epoch >= 0) {
			for (int i = 0; i < nValidationRecords; ++i) {
				layer0->Input2Output(inputs_validation[i], models0, true);
				layer1->Input2Output(models0, models1, true);
				layer2->Input2Output(models1, models2, true);
				model_validation[i] = models2[0];
			}
			double training_pearson = Helper::Pearson(model_training, target_training, nTrainingRecords);
			double validation_pearson =
				Helper::Pearson(model_validation, target_validation, nValidationRecords);
			current_time = clock();
			printf("E %d, training %6.3f, validation %6.3f, time %2.3f\n",
				epoch + 1, training_pearson, validation_pearson, 
				(double)(current_time - start_application) / CLOCKS_PER_SEC);
		}
	}
}

void Triangulars_2_Layers() {
	printf("Target is areas of triangulars given by coordinates of vertices, model is classic KAR\n");
	int nFeatures = 6;
	int nTrainingRecords = 10000;
	int nValidationRecords = 2000;
	auto inputs_training = MakeRandomMatrix(nTrainingRecords, nFeatures, 0.0, 100.0);
	auto inputs_validation = MakeRandomMatrix(nValidationRecords, nFeatures, 0.0, 100.0);
	auto target_training = ComputeAreas(inputs_training, nTrainingRecords);
	auto target_validation = ComputeAreas(inputs_validation, nValidationRecords);

	//data is ready, we start training
	clock_t start_application = clock();

	std::vector<double> xmin;
	std::vector<double> xmax;
	Helper::FindMinMaxMatrix(xmin, xmax, inputs_training, nTrainingRecords, nFeatures);
	double targetMin = 0.0;
	double targetMax = 100.0 * 100.0 / 2.0;

	int nU0 = 30;
	int nU1 = 1;
	std::vector<double> tmin0;
	std::vector<double> tmax0;
	for (int i = 0; i < nU0; ++i) {
		tmin0.push_back(targetMin);
		tmax0.push_back(targetMax);
	}

	std::vector<double> tmin1;
	std::vector<double> tmax1;
	for (int i = 0; i < nU1; ++i) {
		tmin1.push_back(targetMin);
		tmax1.push_back(targetMax);
	}

	auto layer0 = std::make_unique<Layer>(nU0, xmin, xmax, tmin0, tmax0, 5);
	auto layer1 = std::make_unique<Layer>(nU1, tmin0, tmax0, tmin1, tmax1, 20);
	auto models0 = std::make_unique<double[]>(nU0);
	auto models1 = std::make_unique<double[]>(nU1);
	auto deltas1 = std::make_unique<double[]>(nU1);
	auto deltas0 = std::make_unique<double[]>(nU0);
	double mu = 0.01;
	for (int epoch = 0; epoch < 200; ++epoch) {
		double error = 0.0;
		for (int i = 0; i < nTrainingRecords; ++i) {
			layer0->Input2Output(inputs_training[i], models0);
			layer1->Input2Output(models0, models1);
			deltas1[0] = target_training[i] - models1[0];
			layer1->GetDeltas(deltas1, deltas0);
			layer1->UpdateLayerUsingMemory(deltas1, mu);
			layer0->UpdateLayerUsingMemory(deltas0, mu);
			error += (target_training[i] - models1[0]) * (target_training[i] - models1[0]);
		}
		error /= nTrainingRecords;
		error = sqrt(error);
		error /= (targetMax - targetMin);
		printf("Epoch %d, current relative error %f\r", epoch, error);
	}

	clock_t end_PWL_training = clock();
	printf("\nTime for training %2.3f sec.\n", (double)(end_PWL_training - start_application) / CLOCKS_PER_SEC);

	//validation
	double error = 0.0;
	for (int i = 0; i < nValidationRecords; ++i) {
		layer0->Input2Output(inputs_validation[i], models0, true);
		layer1->Input2Output(models0, models1, true);
		double residual = target_validation[i] - models1[0];
		error += residual * residual;
	}
	error /= nValidationRecords;
	error = sqrt(error) / (targetMax - targetMin);
	printf("Relative RMSE error for unseen data %5.5f\n\n", error);
}

void Triangulars_3_Layers() {
	printf("Target is areas of triangulars given by coordinates of vertices, model 3 layer KAN\n");
	int nFeatures = 6;
	int nTrainingRecords = 10000;
	int nValidationRecords = 2000;
	auto inputs_training = MakeRandomMatrix(nTrainingRecords, nFeatures, 0.0, 100.0);
	auto inputs_validation = MakeRandomMatrix(nValidationRecords, nFeatures, 0.0, 100.0);
	auto target_training = ComputeAreas(inputs_training, nTrainingRecords);
	auto target_validation = ComputeAreas(inputs_validation, nValidationRecords);

	//data is ready, we start training
	clock_t start_application = clock();

	std::vector<double> xmin;
	std::vector<double> xmax;
	Helper::FindMinMaxMatrix(xmin, xmax, inputs_training, nTrainingRecords, nFeatures);
	double targetMin = 0.0;
	double targetMax = 100.0 * 100.0 / 2.0;

	int nU0 = 30;
	int nU1 = 8;
	int nU2 = 1;

	std::vector<double> tmin0;
	std::vector<double> tmax0;
	for (int i = 0; i < nU0; ++i) {
		tmin0.push_back(targetMin);
		tmax0.push_back(targetMax);
	}

	std::vector<double> tmin1;
	std::vector<double> tmax1;
	for (int i = 0; i < nU1; ++i) {
		tmin1.push_back(targetMin);
		tmax1.push_back(targetMax);
	}

	std::vector<double> tmin2;
	std::vector<double> tmax2;
	for (int i = 0; i < nU2; ++i) {
		tmin2.push_back(targetMin);
		tmax2.push_back(targetMax);
	}

	auto layer0 = std::make_unique<Layer>(nU0, xmin, xmax, tmin0, tmax0, 3);
	auto layer1 = std::make_unique<Layer>(nU1, tmin0, tmax0, tmin1, tmax1, 8);
	auto layer2 = std::make_unique<Layer>(nU2, tmin1, tmax1, tmin2, tmax2, 24);

	auto models0 = std::make_unique<double[]>(nU0);
	auto models1 = std::make_unique<double[]>(nU1);
	auto models2 = std::make_unique<double[]>(nU2);
	auto deltas0 = std::make_unique<double[]>(nU0);
	auto deltas1 = std::make_unique<double[]>(nU1);
	auto deltas2 = std::make_unique<double[]>(nU2);
	double mu = 0.01;
	for (int epoch = 0; epoch < 200; ++epoch) {
		double error = 0.0;
		for (int i = 0; i < nTrainingRecords; ++i) {
			layer0->Input2Output(inputs_training[i], models0);
			layer1->Input2Output(models0, models1);
			layer2->Input2Output(models1, models2);
			deltas2[0] = target_training[i] - models2[0];
			layer2->GetDeltas(deltas2, deltas1);
			layer1->GetDeltas(deltas1, deltas0);
			layer2->UpdateLayerUsingMemory(deltas2, mu);
			layer1->UpdateLayerUsingMemory(deltas1, mu);
			layer0->UpdateLayerUsingMemory(deltas0, mu);
			error += (target_training[i] - models2[0]) * (target_training[i] - models2[0]);
		}
		error /= nTrainingRecords;
		error = sqrt(error);
		error /= (targetMax - targetMin);
		printf("Epoch %d, current relative error %f\r", epoch, error);
	}

	clock_t end_PWL_training = clock();
	printf("\nTime for training %2.3f sec.\n", (double)(end_PWL_training - start_application) / CLOCKS_PER_SEC);

	//validation
	double error = 0.0;
	for (int i = 0; i < nValidationRecords; ++i) {
		layer0->Input2Output(inputs_validation[i], models0, true);
		layer1->Input2Output(models0, models1, true);
		layer2->Input2Output(models1, models2, true);
		double residual = target_validation[i] - models2[0];
		error += residual * residual;
	}
	error /= nValidationRecords;
	error = sqrt(error) / (targetMax - targetMin);
	printf("Relative RMSE error for unseen data %5.5f\n\n", error);
}

int main()
{
	srand((unsigned int)time(NULL));
	Triangulars_2_Layers();
	Triangulars_3_Layers();
	Determinant_4_4_3Layers();
}

