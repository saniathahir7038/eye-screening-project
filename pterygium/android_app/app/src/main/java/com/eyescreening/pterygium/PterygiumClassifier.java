package com.eyescreening.pterygium;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

final class PterygiumClassifier implements AutoCloseable {
    static final int INPUT_SIZE = 224;
    static final String MODEL_ASSET = "pterygium_model.tflite";
    static final String PARAMETER_ASSET = "pterygium_gradcam.bin";
    static final int CHANNELS = 1280;
    private static final float MASK_FRACTION = 0.20f;
    private final Interpreter interpreter;
    private final GradCamMath.Parameters gradCamParameters;
    private final int scoreOutputIndex;
    private final int convolutionOutputIndex;

    static final class Explanation {
        final float score;
        final Bitmap overlay;

        Explanation(float score, Bitmap overlay) {
            this.score = score;
            this.overlay = overlay;
        }
    }

    PterygiumClassifier(AssetManager assets) throws IOException {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(2);
        interpreter = new Interpreter(loadModel(assets), options);
        int[] inputShape = interpreter.getInputTensor(0).shape();
        if (inputShape.length != 4 || inputShape[0] != 1 || inputShape[1] != INPUT_SIZE
                || inputShape[2] != INPUT_SIZE || inputShape[3] != 3
                || interpreter.getOutputTensorCount() != 2) {
            interpreter.close();
            throw new IOException("Unexpected TensorFlow Lite model input or output shape");
        }
        int scoreIndex = -1;
        int convolutionIndex = -1;
        for (int index = 0; index < 2; index++) {
            int[] shape = interpreter.getOutputTensor(index).shape();
            if (shape.length == 2 && shape[0] == 1 && shape[1] == 1) {
                scoreIndex = index;
            } else if (shape.length == 4 && shape[0] == 1 && shape[1] == GradCamMath.SIDE
                    && shape[2] == GradCamMath.SIDE && shape[3] == CHANNELS) {
                convolutionIndex = index;
            }
        }
        if (scoreIndex < 0 || convolutionIndex < 0) {
            interpreter.close();
            throw new IOException("Unexpected explainability model outputs");
        }
        scoreOutputIndex = scoreIndex;
        convolutionOutputIndex = convolutionIndex;
        try {
            gradCamParameters = loadParameters(assets);
        } catch (IOException error) {
            interpreter.close();
            throw error;
        }
    }

    private static MappedByteBuffer loadModel(AssetManager assets) throws IOException {
        try (AssetFileDescriptor descriptor = assets.openFd(MODEL_ASSET);
             FileInputStream input = new FileInputStream(descriptor.getFileDescriptor());
             FileChannel channel = input.getChannel()) {
            return channel.map(FileChannel.MapMode.READ_ONLY, descriptor.getStartOffset(), descriptor.getDeclaredLength());
        }
    }

    private static GradCamMath.Parameters loadParameters(AssetManager assets) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        try (InputStream stream = assets.open(PARAMETER_ASSET)) {
            byte[] chunk = new byte[8192];
            int count;
            while ((count = stream.read(chunk)) != -1) {
                buffer.write(chunk, 0, count);
            }
        }
        ByteBuffer data = ByteBuffer.wrap(buffer.toByteArray()).order(ByteOrder.LITTLE_ENDIAN);
        if (data.remaining() != 8 + 4 + 4 + 5 * CHANNELS * 4) {
            throw new IOException("Unexpected Grad-CAM parameter size");
        }
        byte[] magic = new byte[8];
        data.get(magic);
        if (!"PGCAM001".equals(new String(magic, java.nio.charset.StandardCharsets.US_ASCII))
                || data.getInt() != CHANNELS) {
            throw new IOException("Grad-CAM parameters do not match this model");
        }
        float epsilon = data.getFloat();
        float[][] arrays = new float[5][CHANNELS];
        for (float[] array : arrays) {
            for (int index = 0; index < CHANNELS; index++) {
                array[index] = data.getFloat();
            }
        }
        return new GradCamMath.Parameters(arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], epsilon);
    }

    float predict(Bitmap source) {
        return run(source, false).score;
    }

    Explanation explain(Bitmap source) {
        return run(source, true);
    }

    private Explanation run(Bitmap source, boolean heatmap) {
        Bitmap processed = preprocess(source);
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        processed.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        ByteBuffer input = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
                .order(ByteOrder.nativeOrder());
        for (int pixel : pixels) {
            input.putFloat(Color.red(pixel));
            input.putFloat(Color.green(pixel));
            input.putFloat(Color.blue(pixel));
        }
        input.rewind();
        float[][] score = new float[1][1];
        float[][][][] convolution = new float[1][GradCamMath.SIDE][GradCamMath.SIDE][CHANNELS];
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(scoreOutputIndex, score);
        outputs.put(convolutionOutputIndex, convolution);
        try {
            interpreter.runForMultipleInputsOutputs(new Object[]{input}, outputs);
            if (!heatmap) {
                return new Explanation(score[0][0], null);
            }
            float[] flat = new float[GradCamMath.SIDE * GradCamMath.SIDE * CHANNELS];
            int position = 0;
            for (int y = 0; y < GradCamMath.SIDE; y++) {
                for (int x = 0; x < GradCamMath.SIDE; x++) {
                    System.arraycopy(convolution[0][y][x], 0, flat, position, CHANNELS);
                    position += CHANNELS;
                }
            }
            float[] map = GradCamMath.heatmap(flat, gradCamParameters);
            return new Explanation(score[0][0], overlay(processed, map));
        } finally {
            processed.recycle();
        }
    }

    private static Bitmap overlay(Bitmap processed, float[] map) {
        int side = GradCamMath.SIDE;
        int[] lowPixels = new int[side * side];
        for (int index = 0; index < lowPixels.length; index++) {
            int level = Math.max(0, Math.min(255, Math.round(map[index] * 255)));
            lowPixels[index] = Color.rgb(level, level, level);
        }
        Bitmap low = Bitmap.createBitmap(lowPixels, side, side, Bitmap.Config.ARGB_8888);
        Bitmap expanded = Bitmap.createScaledBitmap(low, INPUT_SIZE, INPUT_SIZE, true);
        int[] base = new int[INPUT_SIZE * INPUT_SIZE];
        int[] intensity = new int[base.length];
        int[] blended = new int[base.length];
        processed.getPixels(base, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        expanded.getPixels(intensity, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        for (int index = 0; index < base.length; index++) {
            float value = Color.red(intensity[index]) / 255f;
            int red = jet(value, 3);
            int green = jet(value, 2);
            int blue = jet(value, 1);
            blended[index] = Color.rgb(
                    Math.round(0.58f * Color.red(base[index]) + 0.42f * red),
                    Math.round(0.58f * Color.green(base[index]) + 0.42f * green),
                    Math.round(0.58f * Color.blue(base[index]) + 0.42f * blue));
        }
        low.recycle();
        if (expanded != low) {
            expanded.recycle();
        }
        return Bitmap.createBitmap(blended, INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
    }

    private static int jet(float value, int centre) {
        float component = Math.max(0, Math.min(1, 1.5f - Math.abs(4 * value - centre)));
        return Math.round(component * 255);
    }

    static Bitmap preprocess(Bitmap source) {
        Bitmap masked = source.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(masked);
        Paint paint = new Paint();
        paint.setColor(Color.rgb(127, 127, 127));
        int maskWidth = (int) Math.ceil(masked.getWidth() * MASK_FRACTION);
        int maskHeight = (int) Math.ceil(masked.getHeight() * MASK_FRACTION);
        canvas.drawRect(0, 0, maskWidth, maskHeight, paint);
        Bitmap resized = Bitmap.createScaledBitmap(masked, INPUT_SIZE, INPUT_SIZE, true);
        if (masked != resized) {
            masked.recycle();
        }
        return resized;
    }

    @Override
    public void close() {
        interpreter.close();
    }
}
