#include <stdio.h>
float exp_approx(float x, int n_term)
{
    float result = 1.0f;
    float term = 1.0f;

    for (int i = 1; i < n_term; i++) {
        term *= x / i;
        result += term;
    }

    return result;
}
int main(void)
{
    float x = 1.0f;
    int n = 10;

    float approx = exp_approx(x, n);

    printf("exp_approx(%.2f, %d) = %.6f\n", x, n, approx);
    printf("valeur attendue ~ 2.718281\n");

    return 0;
}
