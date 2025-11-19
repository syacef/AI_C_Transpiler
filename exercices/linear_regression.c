#include <stdio.h>
float linear_regression_prediction(float* features, float* thetas, int n_parameters)
{
    float prediction = thetas[0];

    for (int i = 0; i < n_parameters; i++) {
        prediction += thetas[i + 1] * features[i];
    }

    return prediction;
}


int main(void){

    float X[] = {1.0f, 1.0f, 1.0f};
    float theta[] = {0.0f, 1.0f, 1.0f, 1.0f};

    float y_pred = linear_regression_prediction(X, theta, 3);

    printf("Prediction: %.2f\n", y_pred);

    return 0;

}
