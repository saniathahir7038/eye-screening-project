package com.eyescreening.pterygium;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public final class EyeFramingTest {
    @Test
    public void keepsImageWhenDetailIsEvenlyDistributed() {
        double[] energy = new double[128];
        for (int row = 0; row < energy.length; row++) {
            energy[row] = 4;
        }
        assertEquals(1.0, EyeFraming.bottomFractionToKeep(energy), 0.0);
    }

    @Test
    public void removesOnlyLowDetailLowerMargin() {
        double[] energy = new double[128];
        for (int row = 0; row < energy.length; row++) {
            energy[row] = row >= 20 && row <= 95 ? 12 : 1;
        }
        double keep = EyeFraming.bottomFractionToKeep(energy);
        assertTrue(keep >= 0.75);
        assertTrue(keep < 0.90);
    }

    @Test
    public void retainsFullImageWhenDetailTouchesBottom() {
        double[] energy = new double[128];
        for (int row = 0; row < energy.length; row++) {
            energy[row] = row >= 20 ? 12 : 1;
        }
        assertEquals(1.0, EyeFraming.bottomFractionToKeep(energy), 0.0);
    }

    @Test
    public void avoidsExcessiveCrop() {
        double[] energy = new double[128];
        for (int row = 0; row < energy.length; row++) {
            energy[row] = row >= 10 && row <= 50 ? 12 : 1;
        }
        assertEquals(1.0, EyeFraming.bottomFractionToKeep(energy), 0.0);
    }

    @Test
    public void requiresSkinLikeLowerMargin() {
        int[] pixels = new int[128 * 128];
        for (int index = 0; index < pixels.length; index++) {
            pixels[index] = (180 << 16) | (120 << 8) | 80;
        }
        assertTrue(EyeFraming.hasSkinLikeLowerMargin(pixels));
        for (int index = 0; index < pixels.length; index++) {
            pixels[index] = (220 << 16) | (205 << 8) | 198;
        }
        assertTrue(!EyeFraming.hasSkinLikeLowerMargin(pixels));
    }
}
