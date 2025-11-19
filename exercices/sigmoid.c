float sigmoid(float x)
{
    float e_neg_x = exp_approx(-x, 10);
    return 1.0f / (1.0f + e_neg_x);
}
