package com.eyescreening.pterygium;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public class PterygiumDecisionTest {
    @Test
    public void thresholdIsInclusive() {
        assertFalse(PterygiumDecision.isSuspected(PterygiumDecision.THRESHOLD - 0.000001f));
        assertTrue(PterygiumDecision.isSuspected(PterygiumDecision.THRESHOLD));
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsInvalidScore() {
        PterygiumDecision.isSuspected(Float.NaN);
    }

    @Test
    public void returnsSafeScreeningLabels() {
        assertEquals("No visible pterygium pattern", PterygiumDecision.label(0.01f));
        assertTrue(PterygiumDecision.label(0.99f).startsWith("Suspected pterygium"));
    }
}
