//Concept: Andrew Polar and Mike Poluektov
//Developer Andrew Polar

//License
//In case if end user finds the way of making a profit by using this code and earns
//billions of US dollars and meet developer bagging change in the street near McDonalds,
//he or she is not in obligation to buy him a sandwich.

//Symmetricity
//In case developer became rich and famous by publishing this code and meet misfortunate
//end user who went bankrupt by using this code, he is also not in obligation to buy
//end user a sandwich.

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://arxiv.org/abs/2305.08194

#include <iostream>
#include <vector>
#include "Helper.h"
#include "Layer.h"

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

void Triangulars_2_Layers() {
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
}

