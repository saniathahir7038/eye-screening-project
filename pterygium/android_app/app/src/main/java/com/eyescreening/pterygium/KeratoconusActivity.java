package com.eyescreening.pterygium;

import android.os.Bundle;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import java.util.Locale;

public final class KeratoconusActivity extends AppCompatActivity {
    private Spinner sampleSpinner;
    private ImageView sampleImage;
    private ImageView axialMap;
    private ImageView tangentialMap;
    private TextView resultText;
    private TextView metricsText;
    private View resultContainer;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_keratoconus);

        sampleSpinner = findViewById(R.id.sampleSpinner);
        sampleImage = findViewById(R.id.sampleImage);
        axialMap = findViewById(R.id.axialMap);
        tangentialMap = findViewById(R.id.tangentialMap);
        resultText = findViewById(R.id.keratoconusResultText);
        metricsText = findViewById(R.id.keratoconusMetricsText);
        resultContainer = findViewById(R.id.datasetResultContainer);

        String[] names = new String[KeratoconusSample.ALL.length];
        for (int index = 0; index < names.length; index++) {
            names[index] = KeratoconusSample.ALL[index].displayName;
        }
        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, names);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        sampleSpinner.setAdapter(adapter);

        findViewById(R.id.runSampleButton).setOnClickListener(view -> showSelectedSample());
        findViewById(R.id.backToModulesButton).setOnClickListener(view -> finish());
    }

    private void showSelectedSample() {
        KeratoconusSample sample = KeratoconusSample.ALL[sampleSpinner.getSelectedItemPosition()];
        KeratoconusDecision.Outcome outcome = KeratoconusDecision.classify(sample.ppk);

        sampleImage.setImageResource(sample.imageResource);
        axialMap.setImageResource(sample.axialMapResource);
        tangentialMap.setImageResource(sample.tangentialMapResource);

        int resultLabel;
        int resultColour;
        if (outcome == KeratoconusDecision.Outcome.NO_PATTERN) {
            resultLabel = R.string.result_no_kc;
            resultColour = R.color.result_normal;
        } else if (outcome == KeratoconusDecision.Outcome.POSSIBLE) {
            resultLabel = R.string.result_possible_kc;
            resultColour = R.color.result_suspected;
        } else {
            resultLabel = R.string.result_suspected_kc;
            resultColour = R.color.result_suspected;
        }

        resultText.setText(resultLabel);
        resultText.setTextColor(ContextCompat.getColor(this, resultColour));
        metricsText.setText(String.format(
                Locale.US,
                "Dataset label: %s\nPPK: %.0f%%  •  aCLMI: %.2f\nSim-K: %.2f D / %.2f D  •  Difference: %.2f D",
                sample.datasetLabel,
                sample.ppk * 100.0,
                sample.aClmi,
                sample.simK1,
                sample.simK2,
                Math.abs(sample.simK1 - sample.simK2)));
        resultContainer.setVisibility(View.VISIBLE);
    }
}
