package com.eyescreening.pterygium;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class KeratoconusActivity extends AppCompatActivity {
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private Spinner sampleSpinner;
    private ImageView sampleImage;
    private ImageView axialMap;
    private ImageView tangentialMap;
    private TextView resultText;
    private TextView metricsText;
    private View resultContainer;
    private ProgressBar progress;
    private PlacidoSegmentationEngine engine;
    private Uri cameraUri;

    private final ActivityResultLauncher<String> imagePicker = registerForActivityResult(
            new ActivityResultContracts.GetContent(), uri -> {
                if (uri != null) analyseLive(uri);
            });

    private final ActivityResultLauncher<Intent> cameraCapture = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(), result -> {
                if (result.getResultCode() == Activity.RESULT_OK && cameraUri != null) {
                    analyseLive(cameraUri);
                }
            });

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
        progress = findViewById(R.id.keratoconusProgress);

        String[] names = new String[KeratoconusSample.ALL.length];
        for (int index = 0; index < names.length; index++) names[index] = KeratoconusSample.ALL[index].displayName;
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, names);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        sampleSpinner.setAdapter(adapter);

        findViewById(R.id.runSampleButton).setOnClickListener(view -> showSelectedSample());
        findViewById(R.id.captureKeratoconusButton).setOnClickListener(view -> capturePhoto());
        findViewById(R.id.chooseKeratoconusButton).setOnClickListener(view -> imagePicker.launch("image/*"));
        findViewById(R.id.backToModulesButton).setOnClickListener(view -> finish());
        setBusy(true);
        executor.execute(() -> {
            try {
                engine = new PlacidoSegmentationEngine(this);
                runOnUiThread(() -> setBusy(false));
            } catch (Exception ignored) {
                runOnUiThread(() -> showFailure("The Placido model could not be loaded. Reinstall the APK."));
            }
        });
    }

    private void capturePhoto() {
        try {
            File directory = new File(getCacheDir(), "placido-images");
            if (!directory.exists() && !directory.mkdirs()) throw new IOException("cache");
            File file = File.createTempFile("placido-", ".jpg", directory);
            cameraUri = FileProvider.getUriForFile(this, BuildConfig.APPLICATION_ID + ".files", file);
            Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            intent.putExtra(MediaStore.EXTRA_OUTPUT, cameraUri);
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            cameraCapture.launch(intent);
        } catch (Exception error) {
            showFailure("The camera could not be opened. Choose a saved Placido photo instead.");
        }
    }

    private void analyseLive(Uri uri) {
        setBusy(true);
        executor.execute(() -> {
            try (InputStream input = getContentResolver().openInputStream(uri)) {
                Bitmap bitmap = BitmapFactory.decodeStream(input);
                if (bitmap == null || engine == null) throw new IOException("model");
                PlacidoSegmentationEngine.Result result = engine.analyse(bitmap);
                runOnUiThread(() -> showLiveResult(result));
            } catch (Exception error) {
                runOnUiThread(() -> showFailure("This image could not be analysed. Retake a sharp, centred Placido photo."));
            }
        });
    }

    private void showLiveResult(PlacidoSegmentationEngine.Result result) {
        setBusy(false);
        resultContainer.setVisibility(View.VISIBLE);
        sampleImage.setImageBitmap(result.overlay);
        axialMap.setVisibility(View.GONE);
        tangentialMap.setVisibility(View.GONE);
        if (!result.accepted) {
            resultText.setText("Inconclusive - retake required");
            resultText.setTextColor(ContextCompat.getColor(this, R.color.result_suspected));
            metricsText.setText(result.failureReason);
            return;
        }
        boolean irregular = result.irregularity >= 0.040;
        resultText.setText(irregular
                ? "Possible keratoconus pattern - examination recommended"
                : "No keratoconus pattern detected");
        resultText.setTextColor(ContextCompat.getColor(this,
                irregular ? R.color.result_suspected : R.color.result_normal));
        metricsText.setText(String.format(Locale.US,
                "Live experimental analysis\nRings detected: %d of 14\nRing irregularity: %.3f\nCentre: %.0f, %.0f\n\nCompare this capture with the desktop SmartKC result during hardware validation.",
                result.detectedRings, result.irregularity, result.centerX, result.centerY));
    }

    private void showSelectedSample() {
        KeratoconusSample sample = KeratoconusSample.ALL[sampleSpinner.getSelectedItemPosition()];
        setBusy(true);
        executor.execute(() -> {
            try {
                Bitmap bitmap = BitmapFactory.decodeResource(getResources(), sample.imageResource);
                PlacidoSegmentationEngine.Result mobile = engine.analyse(bitmap);
                runOnUiThread(() -> showSelectedSampleResult(sample, mobile));
            } catch (Exception error) {
                runOnUiThread(() -> showFailure("The sample could not be processed by the mobile Placido model."));
            }
        });
    }

    private void showSelectedSampleResult(KeratoconusSample sample,
                                          PlacidoSegmentationEngine.Result mobile) {
        setBusy(false);
        KeratoconusDecision.Outcome outcome = KeratoconusDecision.classify(sample.ppk);
        sampleImage.setImageResource(sample.imageResource);
        axialMap.setVisibility(View.VISIBLE);
        tangentialMap.setVisibility(View.VISIBLE);
        axialMap.setImageResource(sample.axialMapResource);
        tangentialMap.setImageResource(sample.tangentialMapResource);
        int label = outcome == KeratoconusDecision.Outcome.NO_PATTERN ? R.string.result_no_kc
                : outcome == KeratoconusDecision.Outcome.POSSIBLE ? R.string.result_possible_kc
                : R.string.result_suspected_kc;
        resultText.setText(label);
        resultText.setTextColor(ContextCompat.getColor(this,
                outcome == KeratoconusDecision.Outcome.NO_PATTERN ? R.color.result_normal : R.color.result_suspected));
        metricsText.setText(String.format(Locale.US,
                "Saved reference result\nDataset label: %s\nPPK: %.0f%%  |  aCLMI: %.2f\nSim-K: %.2f D / %.2f D  |  Difference: %.2f D\n\nMobile model check: %s, %d rings, irregularity %.3f",
                sample.datasetLabel, sample.ppk * 100.0, sample.aClmi,
                sample.simK1, sample.simK2, Math.abs(sample.simK1 - sample.simK2),
                mobile.accepted ? "passed" : "failed", mobile.detectedRings, mobile.irregularity));
        resultContainer.setVisibility(View.VISIBLE);
    }

    private void setBusy(boolean busy) {
        progress.setVisibility(busy ? View.VISIBLE : View.GONE);
        boolean enabled = !busy && engine != null;
        findViewById(R.id.captureKeratoconusButton).setEnabled(enabled);
        findViewById(R.id.chooseKeratoconusButton).setEnabled(enabled);
        findViewById(R.id.runSampleButton).setEnabled(enabled);
    }

    private void showFailure(String message) {
        setBusy(false);
        resultContainer.setVisibility(View.VISIBLE);
        resultText.setText("Inconclusive");
        resultText.setTextColor(ContextCompat.getColor(this, R.color.result_suspected));
        metricsText.setText(message);
    }

    @Override
    protected void onDestroy() {
        executor.shutdownNow();
        super.onDestroy();
    }
}
