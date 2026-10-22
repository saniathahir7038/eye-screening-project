package com.eyescreening.pterygium;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

final class AssetFiles {
    private AssetFiles() {}

    static String copyToFiles(Context context, String assetName) throws IOException {
        File target = new File(context.getFilesDir(), assetName);
        if (target.exists() && target.length() > 0) {
            return target.getAbsolutePath();
        }
        try (InputStream input = context.getAssets().open(assetName);
             FileOutputStream output = new FileOutputStream(target)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
        }
        return target.getAbsolutePath();
    }
}
