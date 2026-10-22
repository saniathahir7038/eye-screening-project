package com.eyescreening.pterygium;

import android.app.Activity;
import android.content.ClipData;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.MediaController;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.VideoView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;

import java.io.File;
import java.io.IOException;

public final class DryEyeActivity extends AppCompatActivity {
    private static final String STATE_VIDEO_URI = "dry_eye_video_uri";
    private static final int MAX_RECORDING_SECONDS = 15;

    private Spinner eyeSpinner;
    private VideoView videoPreview;
    private TextView statusText;
    private Button shareButton;
    private Uri capturedVideoUri;

    private final ActivityResultLauncher<Intent> videoCapture = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == Activity.RESULT_OK && capturedVideoUri != null) {
                    showCapturedVideo();
                } else {
                    capturedVideoUri = null;
                    statusText.setText(R.string.dry_eye_capture_failed);
                    shareButton.setVisibility(View.GONE);
                }
            });

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_dry_eye);

        eyeSpinner = findViewById(R.id.dryEyeSideSpinner);
        videoPreview = findViewById(R.id.dryEyeVideoPreview);
        statusText = findViewById(R.id.dryEyeStatusText);
        shareButton = findViewById(R.id.shareDryEyeVideoButton);

        ArrayAdapter<CharSequence> eyeAdapter = ArrayAdapter.createFromResource(
                this,
                R.array.eye_sides,
                android.R.layout.simple_spinner_item);
        eyeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        eyeSpinner.setAdapter(eyeAdapter);

        MediaController controller = new MediaController(this);
        controller.setAnchorView(videoPreview);
        videoPreview.setMediaController(controller);

        if (savedInstanceState != null) {
            String savedUri = savedInstanceState.getString(STATE_VIDEO_URI);
            if (savedUri != null) {
                capturedVideoUri = Uri.parse(savedUri);
                showCapturedVideo();
            }
        }

        findViewById(R.id.recordDryEyeVideoButton).setOnClickListener(
                view -> recordVideo());
        shareButton.setOnClickListener(view -> shareVideo());
        findViewById(R.id.backFromDryEyeButton).setOnClickListener(view -> finish());
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        if (capturedVideoUri != null) {
            outState.putString(STATE_VIDEO_URI, capturedVideoUri.toString());
        }
    }

    private void recordVideo() {
        try {
            File directory = new File(getCacheDir(), "captured-eye-videos");
            if (!directory.exists() && !directory.mkdirs()) {
                throw new IOException("Could not create video directory");
            }

            String eye = eyeSpinner.getSelectedItemPosition() == 0 ? "left" : "right";
            File videoFile = File.createTempFile("dry-eye-" + eye + "-", ".mp4", directory);
            capturedVideoUri = FileProvider.getUriForFile(
                    this, BuildConfig.APPLICATION_ID + ".files", videoFile);

            Intent intent = new Intent(MediaStore.ACTION_VIDEO_CAPTURE);
            intent.putExtra(MediaStore.EXTRA_OUTPUT, capturedVideoUri);
            intent.putExtra(MediaStore.EXTRA_DURATION_LIMIT, MAX_RECORDING_SECONDS);
            intent.putExtra(MediaStore.EXTRA_VIDEO_QUALITY, 1);
            intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                    | Intent.FLAG_GRANT_READ_URI_PERMISSION);

            if (intent.resolveActivity(getPackageManager()) == null) {
                throw new IOException("No video camera is available");
            }
            videoCapture.launch(intent);
        } catch (IOException error) {
            capturedVideoUri = null;
            statusText.setText(R.string.dry_eye_capture_failed);
            shareButton.setVisibility(View.GONE);
        }
    }

    private void showCapturedVideo() {
        videoPreview.setVideoURI(capturedVideoUri);
        videoPreview.setVisibility(View.VISIBLE);
        statusText.setText(R.string.dry_eye_video_ready);
        shareButton.setVisibility(View.VISIBLE);
    }

    private void shareVideo() {
        if (capturedVideoUri == null) {
            return;
        }
        Intent share = new Intent(Intent.ACTION_SEND);
        share.setType("video/mp4");
        share.putExtra(Intent.EXTRA_STREAM, capturedVideoUri);
        share.setClipData(ClipData.newRawUri("Dry Eye hardware test video", capturedVideoUri));
        share.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivity(Intent.createChooser(share, getString(R.string.share_test_video)));
    }
}
