package com.eyescreening.pterygium;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.net.Uri;

import java.util.ArrayList;
import java.util.List;

final class DryEyeVideoAnalyzer {
    enum State { NO_INSTABILITY, SUSPECTED, INCONCLUSIVE }

    static final class Result {
        final State state;
        final double nibutSeconds;
        final double blinkSeconds;
        final double analysedSeconds;
        final String detail;

        Result(State state, double nibutSeconds, double blinkSeconds,
               double analysedSeconds, String detail) {
            this.state = state;
            this.nibutSeconds = nibutSeconds;
            this.blinkSeconds = blinkSeconds;
            this.analysedSeconds = analysedSeconds;
            this.detail = detail;
        }
    }

    private DryEyeVideoAnalyzer() {}

    static Result analyse(Context context, Uri uri) {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        try {
            retriever.setDataSource(context, uri);
            String durationValue = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
            long durationMs = durationValue == null ? 0 : Long.parseLong(durationValue);
            if (durationMs < 2500) {
                return inconclusive(durationMs / 1000.0, "Video is too short. Record one blink followed by at least 10 seconds with the eye open.");
            }
            long stepMs = 250;
            List<Double> means = new ArrayList<>();
            List<Double> edges = new ArrayList<>();
            for (long time = 0; time <= durationMs; time += stepMs) {
                Bitmap frame = retriever.getFrameAtTime(time * 1000,
                        MediaMetadataRetriever.OPTION_CLOSEST);
                if (frame == null) continue;
                Bitmap small = Bitmap.createScaledBitmap(frame, 160, 120, true);
                double sum = 0;
                double edge = 0;
                int[] row = new int[160 * 120];
                small.getPixels(row, 0, 160, 0, 0, 160, 120);
                double[] gray = new double[row.length];
                for (int i = 0; i < row.length; i++) {
                    int pixel = row[i];
                    gray[i] = (0.299 * ((pixel >> 16) & 255)
                            + 0.587 * ((pixel >> 8) & 255) + 0.114 * (pixel & 255));
                    sum += gray[i];
                }
                for (int y = 1; y < 119; y++) {
                    for (int x = 1; x < 159; x++) {
                        int i = y * 160 + x;
                        edge += Math.abs(gray[i] - gray[i - 1]) + Math.abs(gray[i] - gray[i - 160]);
                    }
                }
                means.add(sum / gray.length);
                edges.add(edge / (158.0 * 118.0 * 2.0));
                if (small != frame) small.recycle();
                frame.recycle();
            }
            if (means.size() < 10) return inconclusive(durationMs / 1000.0, "Not enough video frames could be decoded.");

            int blink = -1;
            double blinkChange = 0;
            for (int i = 1; i < means.size(); i++) {
                double change = Math.abs(means.get(i) - means.get(i - 1));
                if (change > blinkChange) {
                    blinkChange = change;
                    blink = i;
                }
            }
            if (blink < 0 || blinkChange < 8.0) {
                return inconclusive(durationMs / 1000.0,
                        "No clear blink was detected. Keep the rings centred, blink once, then hold the eye open.");
            }
            int baselineStart = Math.min(edges.size() - 1, blink + 2);
            int baselineEnd = Math.min(edges.size(), baselineStart + 5);
            if (baselineEnd - baselineStart < 3) return inconclusive(durationMs / 1000.0, "Too little open-eye video followed the blink.");
            double baseline = 0;
            for (int i = baselineStart; i < baselineEnd; i++) baseline += edges.get(i);
            baseline /= baselineEnd - baselineStart;
            if (baseline < 4.0) return inconclusive(durationMs / 1000.0, "Placido rings were not sharp enough for tear-film analysis.");

            int sustained = 0;
            int breakIndex = -1;
            for (int i = baselineEnd; i < edges.size(); i++) {
                if (edges.get(i) < baseline * 0.78) sustained++; else sustained = 0;
                if (sustained >= 3) {
                    breakIndex = i - 2;
                    break;
                }
            }
            double blinkSeconds = blink * 0.25;
            if (breakIndex >= 0) {
                double nibut = Math.max(0, breakIndex * 0.25 - blinkSeconds);
                return new Result(nibut < 10 ? State.SUSPECTED : State.NO_INSTABILITY,
                        nibut, blinkSeconds, durationMs / 1000.0,
                        "A sustained reduction in the reflected-ring signal was detected.");
            }
            double observed = durationMs / 1000.0 - blinkSeconds;
            if (observed < 10) return inconclusive(durationMs / 1000.0,
                    "No break was found, but less than 10 seconds after the blink was available.");
            return new Result(State.NO_INSTABILITY, observed, blinkSeconds,
                    durationMs / 1000.0, "No sustained ring break was found during the observed period.");
        } catch (Exception error) {
            return inconclusive(0, "The video could not be analysed on this phone.");
        } finally {
            try {
                retriever.release();
            } catch (Exception ignored) {
                // The analysis result is still valid when decoder cleanup reports an error.
            }
        }
    }

    private static Result inconclusive(double duration, String detail) {
        return new Result(State.INCONCLUSIVE, Double.NaN, Double.NaN, duration, detail);
    }
}
