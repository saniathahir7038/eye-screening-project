package com.eyescreening.pterygium;

final class KeratoconusSample {
    static final KeratoconusSample[] ALL = {
            new KeratoconusSample("No-KC sample — left eye", "No keratoconus",
                    R.drawable.nokc_left, R.drawable.nokc_left_axial_map,
                    R.drawable.nokc_left_tangential_map, 41.21, 40.79, 1.49, 0.04),
            new KeratoconusSample("No-KC sample — right eye", "No keratoconus",
                    R.drawable.nokc_right, R.drawable.nokc_right_axial_map,
                    R.drawable.nokc_right_tangential_map, 39.44, 38.28, 2.52, 0.25),
            new KeratoconusSample("KC sample — left eye", "Keratoconus",
                    R.drawable.kc_left, R.drawable.kc_left_axial_map,
                    R.drawable.kc_left_tangential_map, 50.15, 45.30, 6.03, 1.00),
            new KeratoconusSample("KC sample — right eye", "Keratoconus",
                    R.drawable.kc_right, R.drawable.kc_right_axial_map,
                    R.drawable.kc_right_tangential_map, 50.40, 45.28, 4.30, 0.94)
    };

    final String displayName;
    final String datasetLabel;
    final int imageResource;
    final int axialMapResource;
    final int tangentialMapResource;
    final double simK1;
    final double simK2;
    final double aClmi;
    final double ppk;

    private KeratoconusSample(
            String displayName,
            String datasetLabel,
            int imageResource,
            int axialMapResource,
            int tangentialMapResource,
            double simK1,
            double simK2,
            double aClmi,
            double ppk) {
        this.displayName = displayName;
        this.datasetLabel = datasetLabel;
        this.imageResource = imageResource;
        this.axialMapResource = axialMapResource;
        this.tangentialMapResource = tangentialMapResource;
        this.simK1 = simK1;
        this.simK2 = simK2;
        this.aClmi = aClmi;
        this.ppk = ppk;
    }
}
