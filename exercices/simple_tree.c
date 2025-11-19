int simple_tree(float *features, int n_features) {
    int x1_pos = features[0] > 0.0f;
    int x2_pos = features[1] > 0.0f;

    return !(x1_pos || x2_pos);
}