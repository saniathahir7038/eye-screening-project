package com.eyescreening.pterygium;

/** Grad-CAM for Conv_1 -> batch norm -> ReLU6 -> global average pool -> sigmoid. */
final class GradCamMath {
    static final int SIDE = 7;

    static final class Parameters {
        final float[] denseWeight;
        final float[] gamma;
        final float[] beta;
        final float[] mean;
        final float[] variance;
        final float epsilon;

        Parameters(float[] denseWeight, float[] gamma, float[] beta, float[] mean,
                   float[] variance, float epsilon) {
            this.denseWeight = denseWeight;
            this.gamma = gamma;
            this.beta = beta;
            this.mean = mean;
            this.variance = variance;
            this.epsilon = epsilon;
        }
    }

    private GradCamMath() {}

    /** Returns a normalized 7x7 map in row-major order. */
    static float[] heatmap(float[] convolution, Parameters parameters) {
        int channels = parameters.denseWeight.length;
        int positions = SIDE * SIDE;
        if (convolution.length != positions * channels || parameters.gamma.length != channels
                || parameters.beta.length != channels || parameters.mean.length != channels
                || parameters.variance.length != channels) {
            throw new IllegalArgumentException("Grad-CAM tensor dimensions do not match");
        }
        float[] channelWeights = new float[channels];
        for (int channel = 0; channel < channels; channel++) {
            float scale = (float) (parameters.gamma[channel]
                    / Math.sqrt(parameters.variance[channel] + parameters.epsilon));
            int active = 0;
            for (int position = 0; position < positions; position++) {
                float batchNorm = (convolution[position * channels + channel]
                        - parameters.mean[channel]) * scale + parameters.beta[channel];
                if (batchNorm > 0 && batchNorm < 6) {
                    active++;
                }
            }
            // The sigmoid derivative and spatial constants are common positive
            // factors and cancel when the map is normalized.
            channelWeights[channel] = parameters.denseWeight[channel] * scale
                    * active / positions;
        }
        float[] map = new float[positions];
        float maximum = 0;
        for (int position = 0; position < positions; position++) {
            double weighted = 0;
            for (int channel = 0; channel < channels; channel++) {
                weighted += convolution[position * channels + channel] * channelWeights[channel];
            }
            map[position] = (float) Math.max(0, weighted);
            maximum = Math.max(maximum, map[position]);
        }
        if (maximum > 0) {
            for (int position = 0; position < positions; position++) {
                map[position] /= maximum;
            }
        }
        return map;
    }
}
