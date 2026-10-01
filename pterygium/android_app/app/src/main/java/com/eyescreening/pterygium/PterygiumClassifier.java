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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

final class PterygiumClassifier implements AutoCloseable {
    static final int INPUT_SIZE = 224;
    static final String MODEL_ASSET = "pterygium_model.tflite";
    static final String MODEL_SHA256 = "15ada3c60d498648cca005bde6182f7df90375cf30c46ac45d4929c54345511c";
    private static final float MASK_FRACTION = 0.20f;
    private final Interpreter interpreter;

    PterygiumClassifier(AssetManager assets) throws IOException {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(2);
        interpreter = new Interpreter(loadModel(assets), options);
        int[] inputShape = interpreter.getInputTensor(0).shape();
        int[] outputShape = interpreter.getOutputTensor(0).shape();
        if (inputShape.length != 4 || inputShape[0] != 1 || inputShape[1] != INPUT_SIZE
                || inputShape[2] != INPUT_SIZE || inputShape[3] != 3
                || outputShape.length != 2 || outputShape[0] != 1 || outputShape[1] != 1) {
            interpreter.close();
            throw new IOException("Unexpected TensorFlow Lite model input or output shape");
        }
    }

    private static MappedByteBuffer loadModel(AssetManager assets) throws IOException {
        try (AssetFileDescriptor descriptor = assets.openFd(MODEL_ASSET);
             FileInputStream input = new FileInputStream(descriptor.getFileDescriptor());
             FileChannel channel = input.getChannel()) {
            return channel.map(FileChannel.MapMode.READ_ONLY, descriptor.getStartOffset(), descriptor.getDeclaredLength());
        }
    }

    float predict(Bitmap source) {
        Bitmap processed = preprocess(source);
        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        processed.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        processed.recycle();

        ByteBuffer input = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
                .order(ByteOrder.nativeOrder());
        for (int pixel : pixels) {
            input.putFloat(Color.red(pixel));
            input.putFloat(Color.green(pixel));
            input.putFloat(Color.blue(pixel));
        }
        input.rewind();
        float[][] output = new float[1][1];
        interpreter.run(input, output);
        return output[0][0];
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
