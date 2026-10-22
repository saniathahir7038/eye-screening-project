package com.eyescreening.pterygium;

public final class KeratoconusDecision {
    public enum Outcome {
        NO_PATTERN,
        POSSIBLE,
        SUSPECTED
    }

    private KeratoconusDecision() {
    }

    public static Outcome classify(double ppk) {
        if (ppk < 0.0 || ppk > 1.0 || Double.isNaN(ppk)) {
            throw new IllegalArgumentException("PPK must be between 0 and 1");
        }
        if (ppk < 0.20) {
            return Outcome.NO_PATTERN;
        }
        if (ppk < 0.45) {
            return Outcome.POSSIBLE;
        }
        return Outcome.SUSPECTED;
    }
}
