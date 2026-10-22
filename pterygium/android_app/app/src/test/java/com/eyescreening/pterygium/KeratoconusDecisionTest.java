package com.eyescreening.pterygium;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class KeratoconusDecisionTest {
    @Test
    public void classifiesPublishedPpkRanges() {
        assertEquals(KeratoconusDecision.Outcome.NO_PATTERN,
                KeratoconusDecision.classify(0.04));
        assertEquals(KeratoconusDecision.Outcome.POSSIBLE,
                KeratoconusDecision.classify(0.25));
        assertEquals(KeratoconusDecision.Outcome.SUSPECTED,
                KeratoconusDecision.classify(0.45));
        assertEquals(KeratoconusDecision.Outcome.SUSPECTED,
                KeratoconusDecision.classify(0.94));
    }
}
