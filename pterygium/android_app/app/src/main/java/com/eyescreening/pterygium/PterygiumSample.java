package com.eyescreening.pterygium;

final class PterygiumSample {
    static final PterygiumSample[] ALL = {
            new PterygiumSample("Pterygium sample 1609", R.drawable.pterygium_positive_1609),
            new PterygiumSample("Pterygium sample 1658", R.drawable.pterygium_positive_1658),
            new PterygiumSample("No-pterygium sample 160", R.drawable.pterygium_negative_160),
            new PterygiumSample("No-pterygium sample 73", R.drawable.pterygium_negative_73)
    };
    final String name;
    final int resource;
    private PterygiumSample(String name, int resource) {
        this.name = name;
        this.resource = resource;
    }
}
