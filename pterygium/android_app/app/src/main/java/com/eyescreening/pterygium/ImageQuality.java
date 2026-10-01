package com.eyescreening.pterygium;

import android.graphics.Bitmap;
import android.graphics.Color;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

final class ImageQuality {
    static final int CHECK_SIZE = 224;
    static final int MIN_WIDTH = 224;
    static final int MIN_HEIGHT = 224;
    static final double MIN_ASPECT_RATIO = 0.75;
    static final double MAX_ASPECT_RATIO = 2.0;
    static final double MIN_BRIGHTNESS = 45.0;
    static final double MAX_BRIGHTNESS = 210.0;
    static final double MIN_CONTRAST = 18.0;
    static final double MIN_SHARPNESS = 20.0;

    static final class Result {
        final boolean accepted;
        final int width;
        final int height;
        final double aspectRatio;
        final double brightness;
        final double contrast;
        final double sharpness;
        final List<String> issues;

        Result(boolean accepted, int width, int height, double aspectRatio, double brightness,
               double contrast, double sharpness, List<String> issues) {
            this.accepted = accepted;
            this.width = width;
            this.height = height;
            this.aspectRatio = aspectRatio;
            this.brightness = brightness;
            this.contrast = contrast;
            this.sharpness = sharpness;
            this.issues = Collections.unmodifiableList(new ArrayList<>(issues));
        }
    }

    private ImageQuality() {}

    static Result assess(Bitmap source) {
        int width = source.getWidth();
        int height = source.getHeight();
        double aspectRatio = (double) width / height;
        Bitmap resized = Bitmap.createScaledBitmap(source, CHECK_SIZE, CHECK_SIZE, true);
        int[] pixels = new int[CHECK_SIZE * CHECK_SIZE];
        resized.getPixels(pixels, 0, CHECK_SIZE, 0, 0, CHECK_SIZE, CHECK_SIZE);
        if (resized != source) {
            resized.recycle();
        }

        double[] grayscale = new double[pixels.length];
        double sum = 0.0;
        for (int index = 0; index < pixels.length; index++) {
            int pixel = pixels[index];
            double value = 0.299 * Color.red(pixel) + 0.587 * Color.green(pixel) + 0.114 * Color.blue(pixel);
            grayscale[index] = value;
            sum += value;
        }
        double brightness = sum / grayscale.length;
        double squaredDifference = 0.0;
        for (double value : grayscale) {
            double difference = value - brightness;
            squaredDifference += difference * difference;
        }
        double contrast = Math.sqrt(squaredDifference / grayscale.length);

        int laplacianCount = (CHECK_SIZE - 2) * (CHECK_SIZE - 2);
        double laplacianSum = 0.0;
        double laplacianSquareSum = 0.0;
        for (int y = 1; y < CHECK_SIZE - 1; y++) {
            for (int x = 1; x < CHECK_SIZE - 1; x++) {
                int index = y * CHECK_SIZE + x;
                double laplacian = grayscale[index - 1] + grayscale[index + 1]
                        + grayscale[index - CHECK_SIZE] + grayscale[index + CHECK_SIZE]
                        - 4.0 * grayscale[index];
                laplacianSum += laplacian;
                laplacianSquareSum += laplacian * laplacian;
            }
        }
        double laplacianMean = laplacianSum / laplacianCount;
        double sharpness = laplacianSquareSum / laplacianCount - laplacianMean * laplacianMean;

        List<String> issues = new ArrayList<>();
        if (width < MIN_WIDTH || height < MIN_HEIGHT) {
            issues.add("Use an image of at least 224 × 224 pixels.");
        }
        if (aspectRatio < MIN_ASPECT_RATIO || aspectRatio > MAX_ASPECT_RATIO) {
            issues.add("Retake a close-up landscape or near-square eye photograph.");
        }
        if (brightness < MIN_BRIGHTNESS) {
            issues.add("The image is too dark. Use brighter, even lighting.");
        } else if (brightness > MAX_BRIGHTNESS) {
            issues.add("The image is overexposed. Reduce glare and direct light.");
        }
        if (contrast < MIN_CONTRAST) {
            issues.add("The image has too little contrast. Improve lighting and camera focus.");
        }
        if (sharpness < MIN_SHARPNESS) {
            issues.add("The image appears blurred. Hold the camera steady and refocus.");
        }
        return new Result(issues.isEmpty(), width, height, aspectRatio, brightness, contrast, sharpness, issues);
    }
}
