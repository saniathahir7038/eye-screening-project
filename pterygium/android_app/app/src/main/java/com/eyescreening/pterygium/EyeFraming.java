package com.eyescreening.pterygium;

import android.graphics.Bitmap;
import android.graphics.Color;

import java.util.Arrays;

/** Removes only a confidently low-detail strip below the visible eye. */
final class EyeFraming {
    private static final int SIZE = 128;
    private static final int EDGE_LEFT = 13;
    private static final int EDGE_RIGHT = 115;
    private static final double MIN_EDGE_SEPARATION = 1.5;
    private static final double MIN_REMOVAL_FRACTION = 0.12;
    private static final double MIN_RETAINED_FRACTION = 0.75;
    private static final double BOTTOM_MARGIN_FRACTION = 0.12;

    static final class Result {
        final Bitmap image;
        final double removedBottomFraction;

        Result(Bitmap image, double removedBottomFraction) {
            this.image = image;
            this.removedBottomFraction = removedBottomFraction;
        }

        boolean adjusted() {
            return removedBottomFraction > 0;
        }
    }

    private EyeFraming() {}

    static Result frame(Bitmap source) {
        Bitmap preview = Bitmap.createScaledBitmap(source, SIZE, SIZE, true);
        int[] pixels = new int[SIZE * SIZE];
        preview.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE);
        if (preview != source) {
            preview.recycle();
        }

        double[] gray = new double[pixels.length];
        for (int index = 0; index < pixels.length; index++) {
            int pixel = pixels[index];
            gray[index] = 0.299 * Color.red(pixel) + 0.587 * Color.green(pixel)
                    + 0.114 * Color.blue(pixel);
        }
        double[] rowEnergy = new double[SIZE];
        for (int y = 1; y < SIZE - 1; y++) {
            double total = 0;
            for (int x = EDGE_LEFT; x < EDGE_RIGHT; x++) {
                total += Math.abs(gray[(y + 1) * SIZE + x] - gray[(y - 1) * SIZE + x]);
            }
            rowEnergy[y] = total / (EDGE_RIGHT - EDGE_LEFT);
        }
        rowEnergy[0] = rowEnergy[1];
        rowEnergy[SIZE - 1] = rowEnergy[SIZE - 2];
        double[] smoothed = new double[SIZE];
        for (int y = 0; y < SIZE; y++) {
            double total = 0;
            for (int offset = -4; offset <= 4; offset++) {
                int row = Math.max(0, Math.min(SIZE - 1, y + offset));
                total += rowEnergy[row];
            }
            smoothed[y] = total / 9;
        }
        double fraction = bottomFractionToKeep(smoothed);
        if (fraction == 1.0 || !hasSkinLikeLowerMargin(pixels)) {
            return new Result(source, 0);
        }
        int cropHeight = Math.max(1, Math.min(source.getHeight(),
                (int) Math.round(source.getHeight() * fraction)));
        Bitmap framed = Bitmap.createBitmap(source, 0, 0, source.getWidth(), cropHeight);
        return new Result(framed, 1.0 - (double) cropHeight / source.getHeight());
    }

    static double bottomFractionToKeep(double[] energy) {
        if (energy.length != SIZE) {
            throw new IllegalArgumentException("Expected 128 rows of edge energy");
        }
        double[] sorted = energy.clone();
        Arrays.sort(sorted);
        double baseline = sorted[(int) (0.20 * (SIZE - 1))];
        double peak = sorted[(int) (0.90 * (SIZE - 1))];
        if (peak < baseline + MIN_EDGE_SEPARATION) {
            return 1.0;
        }
        double totalActivity = 0;
        for (double value : energy) {
            totalActivity += Math.max(0, value - baseline);
        }
        if (totalActivity <= 0) {
            return 1.0;
        }
        double cumulativeActivity = 0;
        int lastActive = SIZE - 1;
        for (int y = 0; y < SIZE; y++) {
            cumulativeActivity += Math.max(0, energy[y] - baseline);
            if (cumulativeActivity >= 0.95 * totalActivity) {
                lastActive = y;
                break;
            }
        }
        double keep = Math.min(1.0, (lastActive + 1.0) / SIZE + BOTTOM_MARGIN_FRACTION);
        if (keep < MIN_RETAINED_FRACTION || 1.0 - keep < MIN_REMOVAL_FRACTION) {
            return 1.0;
        }
        return keep;
    }

    static boolean hasSkinLikeLowerMargin(int[] pixels) {
        if (pixels.length != SIZE * SIZE) {
            throw new IllegalArgumentException("Expected 128 x 128 pixels");
        }
        double redMinusGreen = 0;
        double greenMinusBlue = 0;
        int count = 0;
        for (int y = 112; y < SIZE; y++) {
            for (int x = EDGE_LEFT; x < EDGE_RIGHT; x++) {
                int pixel = pixels[y * SIZE + x];
                int red = (pixel >>> 16) & 0xff;
                int green = (pixel >>> 8) & 0xff;
                int blue = pixel & 0xff;
                redMinusGreen += red - green;
                greenMinusBlue += green - blue;
                count++;
            }
        }
        return redMinusGreen / count >= 48 && greenMinusBlue / count >= 15;
    }
}
