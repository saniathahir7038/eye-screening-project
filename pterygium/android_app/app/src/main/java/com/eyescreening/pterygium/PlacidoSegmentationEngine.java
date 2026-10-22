package com.eyescreening.pterygium;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.IOException;

final class PlacidoSegmentationEngine {
    static final int SIZE = 512;
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};
    private final Module module;

    static final class Result {
        final boolean accepted;
        final String failureReason;
        final Bitmap crop;
        final Bitmap overlay;
        final float centerX;
        final float centerY;
        final int detectedRings;
        final double irregularity;

        Result(boolean accepted, String failureReason, Bitmap crop, Bitmap overlay,
               float centerX, float centerY, int detectedRings, double irregularity) {
            this.accepted = accepted;
            this.failureReason = failureReason;
            this.crop = crop;
            this.overlay = overlay;
            this.centerX = centerX;
            this.centerY = centerY;
            this.detectedRings = detectedRings;
            this.irregularity = irregularity;
        }
    }

    PlacidoSegmentationEngine(Context context) throws IOException {
        module = LiteModuleLoader.load(AssetFiles.copyToFiles(context, "placido_segmentation.ptl"));
    }

    Result analyse(Bitmap source) {
        Bitmap crop = centreSquare(source);
        Bitmap scaled = Bitmap.createScaledBitmap(crop, SIZE, SIZE, true);
        int[] pixels = new int[SIZE * SIZE];
        scaled.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE);
        float[] values = new float[3 * SIZE * SIZE];
        int plane = SIZE * SIZE;
        for (int i = 0; i < pixels.length; i++) {
            values[i] = (Color.red(pixels[i]) / 255f - MEAN[0]) / STD[0];
            values[plane + i] = (Color.green(pixels[i]) / 255f - MEAN[1]) / STD[1];
            values[2 * plane + i] = (Color.blue(pixels[i]) / 255f - MEAN[2]) / STD[2];
        }
        Tensor output = module.forward(IValue.from(Tensor.fromBlob(
                values, new long[]{1, 3, SIZE, SIZE}))).toTensor();
        float[] logits = output.getDataAsFloatArray();
        int[] labels = new int[plane];
        int[] counts = new int[19];
        for (int i = 0; i < plane; i++) {
            int best = 0;
            float value = logits[i];
            for (int label = 1; label < 19; label++) {
                float candidate = logits[label * plane + i];
                if (candidate > value) {
                    value = candidate;
                    best = label;
                }
            }
            labels[i] = best;
            counts[best]++;
        }

        double cx = 0, cy = 0;
        for (int i = 0; i < plane; i++) {
            if (labels[i] == 1) {
                cx += i % SIZE;
                cy += i / SIZE;
            }
        }
        if (counts[1] < 80) {
            return new Result(false, "Placido centre was not detected. Re-centre the rings and retake.",
                    crop, scaled, Float.NaN, Float.NaN, 0, Double.NaN);
        }
        cx /= counts[1];
        cy /= counts[1];

        int detected = 0;
        double irregularityTotal = 0;
        int usable = 0;
        for (int label = 2; label <= 15; label++) {
            double[] sums = new double[360];
            int[] binCounts = new int[360];
            for (int i = 0; i < plane; i++) {
                if (labels[i] != label) continue;
                double dx = i % SIZE - cx;
                double dy = i / SIZE - cy;
                int angle = (int) Math.floor(Math.toDegrees(Math.atan2(dy, dx)) + 360.0) % 360;
                sums[angle] += Math.hypot(dx, dy);
                binCounts[angle]++;
            }
            int covered = 0;
            double radiusMean = 0;
            for (int angle = 0; angle < 360; angle++) {
                if (binCounts[angle] > 0) {
                    sums[angle] /= binCounts[angle];
                    radiusMean += sums[angle];
                    covered++;
                }
            }
            if (covered >= 180) {
                detected++;
                radiusMean /= covered;
                double variance = 0;
                for (int angle = 0; angle < 360; angle++) {
                    if (binCounts[angle] > 0) {
                        double delta = sums[angle] - radiusMean;
                        variance += delta * delta;
                    }
                }
                irregularityTotal += Math.sqrt(variance / covered) / radiusMean;
                usable++;
            }
        }

        Bitmap overlay = scaled.copy(Bitmap.Config.ARGB_8888, true);
        int[] overlayPixels = pixels.clone();
        int[] palette = {0x00000000, 0xFFE34F4F, 0xFF22C55E, 0xFF38BDF8,
                0xFFF59E0B, 0xFFA78BFA, 0xFFFB7185, 0xFF2DD4BF};
        for (int i = 0; i < plane; i++) {
            int label = labels[i];
            if (label > 0 && (i % 3 == 0)) {
                int tint = palette[1 + ((label - 1) % (palette.length - 1))];
                int original = pixels[i];
                overlayPixels[i] = Color.rgb(
                        (Color.red(original) + Color.red(tint)) / 2,
                        (Color.green(original) + Color.green(tint)) / 2,
                        (Color.blue(original) + Color.blue(tint)) / 2);
            }
        }
        overlay.setPixels(overlayPixels, 0, SIZE, 0, 0, SIZE, SIZE);
        boolean accepted = detected >= 8 && usable > 0;
        String reason = accepted ? "" : "Too few complete Placido rings were detected. Improve focus and alignment.";
        return new Result(accepted, reason, crop, overlay, (float) cx, (float) cy,
                detected, usable == 0 ? Double.NaN : irregularityTotal / usable);
    }

    private static Bitmap centreSquare(Bitmap source) {
        int side = Math.min(source.getWidth(), source.getHeight()) / 3;
        side = Math.max(1, side);
        int left = (source.getWidth() - side) / 2;
        int top = (source.getHeight() - side) / 2;
        return Bitmap.createBitmap(source, left, top, side, side);
    }
}
