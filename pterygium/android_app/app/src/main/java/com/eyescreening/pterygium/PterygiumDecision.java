package com.eyescreening.pterygium;

final class PterygiumDecision {
    static final float THRESHOLD = 0.18100688606500626f;

    private PterygiumDecision() {}

    static boolean isSuspected(float score) {
        if (Float.isNaN(score) || Float.isInfinite(score) || score < 0.0f || score > 1.0f) {
            throw new IllegalArgumentException("Model score must be finite and between zero and one");
        }
        return score >= THRESHOLD;
    }

    static String label(float score) {
        return isSuspected(score)
                ? "Suspected pterygium — professional eye examination recommended"
                : "No visible pterygium pattern";
    }
}
