float logistic_regression(float* features, float* thetas, int n_parameter)
{
    float linear_sum = thetas[0];

    for (int i = 0; i < n_parameter; i++) {
        linear_sum += thetas[i + 1] * features[i];
    }

    return sigmoid(linear_sum);
}
