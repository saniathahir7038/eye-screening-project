package com.eyescreening.pterygium;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.exifinterface.media.ExifInterface;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends AppCompatActivity {
    private static final int MAX_DECODE_DIMENSION = 2048;
    private static final String STATE_CAMERA_URI = "pending_camera_uri";

    private ImageView imagePreview;
    private ImageView heatmapPreview;
    private TextView statusText;
    private TextView resultText;
    private TextView scoreText;
    private ProgressBar progressBar;
    private Button captureButton;
    private Button chooseButton;
    private Button heatmapButton;
    private Bitmap currentHeatmap;
    private final ExecutorService inferenceExecutor = Executors.newSingleThreadExecutor();
    private PterygiumClassifier classifier;
    private Uri pendingCameraUri;

    private final ActivityResultLauncher<String> imagePicker = registerForActivityResult(
            new ActivityResultContracts.GetContent(),
            uri -> {
                if (uri != null) {
                    handleImage(uri);
                }
            });

    private final ActivityResultLauncher<Uri> cameraCapture = registerForActivityResult(
            new ActivityResultContracts.TakePicture(),
            saved -> {
                if (saved && pendingCameraUri != null) {
                    Uri capturedImage = pendingCameraUri;
                    pendingCameraUri = null;
                    handleImage(capturedImage);
                } else {
                    setStatus("Photo capture was cancelled.");
                }
            });

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imagePreview = findViewById(R.id.imagePreview);
        heatmapPreview = findViewById(R.id.heatmapPreview);
        statusText = findViewById(R.id.statusText);
        resultText = findViewById(R.id.resultText);
        scoreText = findViewById(R.id.scoreText);
        progressBar = findViewById(R.id.progressBar);
        captureButton = findViewById(R.id.captureButton);
        chooseButton = findViewById(R.id.chooseButton);
        heatmapButton = findViewById(R.id.heatmapButton);
        if (savedInstanceState != null) {
            String cameraUri = savedInstanceState.getString(STATE_CAMERA_URI);
            if (cameraUri != null) {
                pendingCameraUri = Uri.parse(cameraUri);
            }
        }

        captureButton.setOnClickListener(view -> capturePhoto());
        chooseButton.setOnClickListener(view -> imagePicker.launch("image/*"));
        heatmapButton.setOnClickListener(view -> {
            boolean show = heatmapPreview.getVisibility() != View.VISIBLE;
            heatmapPreview.setVisibility(show ? View.VISIBLE : View.GONE);
            findViewById(R.id.heatmapNotice).setVisibility(show ? View.VISIBLE : View.GONE);
            heatmapButton.setText(show ? R.string.hide_research_heatmap : R.string.show_research_heatmap);
        });
        initialiseClassifier();
    }

    private void initialiseClassifier() {
        setBusy(true, "Loading on-device model…");
        inferenceExecutor.execute(() -> {
            try {
                classifier = new PterygiumClassifier(getAssets());
                runOnUiThread(() -> {
                    setBusy(false, getString(R.string.empty_status));
                    captureButton.setEnabled(true);
                    chooseButton.setEnabled(true);
                });
            } catch (Exception error) {
                runOnUiThread(() -> setBusy(false, "Model could not be loaded. Reinstall this research build."));
            }
        });
    }

    private void capturePhoto() {
        try {
            File directory = new File(getCacheDir(), "captured-eye-images");
            if (!directory.exists() && !directory.mkdirs()) {
                throw new IOException("Could not create the image cache");
            }
            File image = File.createTempFile("eye-", ".jpg", directory);
            pendingCameraUri = FileProvider.getUriForFile(
                    this, BuildConfig.APPLICATION_ID + ".files", image);
            cameraCapture.launch(pendingCameraUri);
        } catch (IOException error) {
            setStatus("The camera could not be opened. Choose a saved photograph instead.");
        }
    }

    private void handleImage(Uri uri) {
        setBusy(true, "Checking image quality…");
        clearHeatmap();
        resultText.setText("");
        scoreText.setText("");
        inferenceExecutor.execute(() -> {
            Bitmap bitmap = null;
            Bitmap original = null;
            try {
                original = decodeBitmap(uri);
                EyeFraming.Result framing = EyeFraming.frame(original);
                bitmap = framing.image;
                Bitmap displayBitmap = bitmap;
                runOnUiThread(() -> imagePreview.setImageBitmap(displayBitmap));

                ImageQuality.Result quality = ImageQuality.assess(bitmap);
                if (!quality.accepted) {
                    String issues = "Image quality insufficient — retake photograph\n\n• "
                            + TextUtils.join("\n• ", quality.issues);
                    runOnUiThread(() -> {
                        setBusy(false, issues);
                        resultText.setText("");
                        scoreText.setText(formatQuality(quality) + formatFraming(framing));
                    });
                    return;
                }
                if (classifier == null) {
                    throw new IOException("The model is not ready");
                }
                float originalScore = framing.adjusted() ? classifier.predict(original) : Float.NaN;
                PterygiumClassifier.Explanation explanation = classifier.explain(bitmap);
                float score = explanation.score;
                boolean suspected = PterygiumDecision.isSuspected(score);
                if (framing.adjusted()
                        && PterygiumDecision.isSuspected(originalScore) != suspected) {
                    explanation.overlay.recycle();
                    runOnUiThread(() -> showInconclusive(originalScore, score, quality, framing));
                } else {
                    runOnUiThread(() -> showResult(score, suspected, quality, framing,
                            explanation.overlay));
                }
            } catch (Exception error) {
                if (bitmap != null && !bitmap.isRecycled()) {
                    bitmap.recycle();
                }
                runOnUiThread(() -> {
                    imagePreview.setImageDrawable(null);
                    setBusy(false, "This file could not be screened. Use a readable PNG or JPEG eye photograph.");
                });
            } finally {
                if (original != null && original != bitmap && !original.isRecycled()) {
                    original.recycle();
                }
            }
        });
    }

    private Bitmap decodeBitmap(Uri uri) throws IOException {
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        try (InputStream stream = getContentResolver().openInputStream(uri)) {
            if (stream == null) {
                throw new IOException("Image could not be opened");
            }
            BitmapFactory.decodeStream(stream, null, bounds);
        }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw new IOException("Image dimensions are invalid");
        }
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        options.inSampleSize = 1;
        while (Math.max(bounds.outWidth / options.inSampleSize, bounds.outHeight / options.inSampleSize)
                > MAX_DECODE_DIMENSION) {
            options.inSampleSize *= 2;
        }
        Bitmap decoded;
        try (InputStream stream = getContentResolver().openInputStream(uri)) {
            decoded = BitmapFactory.decodeStream(stream, null, options);
        }
        if (decoded == null) {
            throw new IOException("Image could not be decoded");
        }
        int orientation = ExifInterface.ORIENTATION_NORMAL;
        try (InputStream stream = getContentResolver().openInputStream(uri)) {
            if (stream != null) {
                orientation = new ExifInterface(stream).getAttributeInt(
                        ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            }
        } catch (IOException ignored) {
            // Missing EXIF metadata is valid for PNG and many camera images.
        }
        return applyOrientation(decoded, orientation);
    }

    private static Bitmap applyOrientation(Bitmap source, int orientation) {
        Matrix matrix = new Matrix();
        switch (orientation) {
            case ExifInterface.ORIENTATION_FLIP_HORIZONTAL:
                matrix.setScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180);
                break;
            case ExifInterface.ORIENTATION_FLIP_VERTICAL:
                matrix.setRotate(180);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_TRANSPOSE:
                matrix.setRotate(90);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90);
                break;
            case ExifInterface.ORIENTATION_TRANSVERSE:
                matrix.setRotate(-90);
                matrix.postScale(-1, 1);
                break;
            case ExifInterface.ORIENTATION_ROTATE_270:
                matrix.setRotate(-90);
                break;
            default:
                return source;
        }
        Bitmap oriented = Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
        if (oriented != source) {
            source.recycle();
        }
        return oriented;
    }

    private void showResult(float score, boolean suspected, ImageQuality.Result quality,
                            EyeFraming.Result framing, Bitmap overlay) {
        setBusy(false, "Experimental research result. Technical checks passed; eye framing still needs visual review.");
        resultText.setText(PterygiumDecision.label(score));
        resultText.setTextColor(ContextCompat.getColor(
                this, suspected ? R.color.result_suspected : R.color.result_normal));
        scoreText.setText(String.format(
                Locale.US,
                "Model score %.3f  |  Decision threshold %.3f\n%s%s",
                score,
                PterygiumDecision.THRESHOLD,
                formatQuality(quality),
                formatFraming(framing)));
        currentHeatmap = overlay;
        heatmapPreview.setImageBitmap(overlay);
        heatmapButton.setVisibility(View.VISIBLE);
    }

    private void showInconclusive(float originalScore, float framedScore,
                                  ImageQuality.Result quality, EyeFraming.Result framing) {
        setBusy(false, "Experimental research result changed after automatic framing.");
        resultText.setText("Unable to screen this photo reliably — retake a closer eye photograph.");
        resultText.setTextColor(ContextCompat.getColor(this, R.color.text_primary));
        scoreText.setText(String.format(Locale.US,
                "Full photo score %.3f  |  Framed photo score %.3f  |  Threshold %.3f\n%s%s",
                originalScore, framedScore, PterygiumDecision.THRESHOLD,
                formatQuality(quality), formatFraming(framing)));
    }

    private static String formatFraming(EyeFraming.Result framing) {
        return framing.adjusted()
                ? String.format(Locale.US, "\nAuto-framing removed %.0f%% of the lower margin. Preview shows the framed area.",
                100 * framing.removedBottomFraction)
                : "\nFull photo used.";
    }

    private static String formatQuality(ImageQuality.Result quality) {
        return String.format(
                Locale.US,
                "Quality: %d × %d, brightness %.1f, contrast %.1f, sharpness %.1f",
                quality.width, quality.height, quality.brightness, quality.contrast, quality.sharpness);
    }

    private void setBusy(boolean busy, String status) {
        progressBar.setVisibility(busy ? View.VISIBLE : View.GONE);
        captureButton.setEnabled(!busy && classifier != null);
        chooseButton.setEnabled(!busy && classifier != null);
        setStatus(status);
    }

    private void setStatus(String status) {
        statusText.setText(status);
    }

    private void clearHeatmap() {
        heatmapPreview.setImageDrawable(null);
        heatmapPreview.setVisibility(View.GONE);
        findViewById(R.id.heatmapNotice).setVisibility(View.GONE);
        heatmapButton.setVisibility(View.GONE);
        heatmapButton.setText(R.string.show_research_heatmap);
        if (currentHeatmap != null) {
            currentHeatmap.recycle();
            currentHeatmap = null;
        }
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        if (pendingCameraUri != null) {
            outState.putString(STATE_CAMERA_URI, pendingCameraUri.toString());
        }
        super.onSaveInstanceState(outState);
    }

    @Override
    protected void onDestroy() {
        clearHeatmap();
        inferenceExecutor.shutdownNow();
        if (classifier != null) {
            classifier.close();
        }
        super.onDestroy();
    }
}
