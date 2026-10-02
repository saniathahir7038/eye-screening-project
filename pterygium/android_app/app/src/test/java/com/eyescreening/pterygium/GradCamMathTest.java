package com.eyescreening.pterygium;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public final class GradCamMathTest {
    @Test
    public void positiveChannelHighlightsActiveLocation() {
        float[] convolution = new float[GradCamMath.SIDE * GradCamMath.SIDE * 2];
        convolution[0] = 2;
        convolution[2] = 1;
        GradCamMath.Parameters parameters = new GradCamMath.Parameters(
                new float[]{1, -1}, new float[]{1, 1}, new float[]{0, 0},
                new float[]{0, 0}, new float[]{1, 1}, 0);
        float[] map = GradCamMath.heatmap(convolution, parameters);
        assertEquals(1, map[0], 0.00001);
        assertEquals(0.5, map[1], 0.00001);
        assertEquals(0, map[2], 0.00001);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsWrongTensorShape() {
        GradCamMath.Parameters parameters = new GradCamMath.Parameters(
                new float[]{1}, new float[]{1}, new float[]{0},
                new float[]{0}, new float[]{1}, 0);
        GradCamMath.heatmap(new float[1], parameters);
    }
}
