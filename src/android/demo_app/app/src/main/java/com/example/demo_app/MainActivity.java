package com.example.demo_app;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;


import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.opencv.core.Mat;

public class MainActivity extends AppCompatActivity implements Runnable {
    private String mImagename = "ants.jpg";

    private ImageView mImageView;
    private Button mButton;
    private TextView mTextView;

    private Bitmap mBitmap = null;
    private Module torchscript_module = null;
    private int argmax = 0;
    private String[] classes = {"Ants", "Bees"};

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (OpenCVLoader.initDebug()) Log.d("LOADED", "success");
        setContentView(R.layout.activity_main);
        try {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            mBitmap = BitmapFactory.decodeStream(getAssets().open(mImagename), null, options);
        } catch (IOException e) {
            Log.e("Demo APP", "Error reading assets", e);
            finish();
        }
        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);
        mTextView = findViewById(R.id.textView);
        mTextView.setText("");

        mButton = findViewById(R.id.Button);
        mButton.setOnClickListener(v -> {
            mButton.setEnabled(false);
            mButton.setText("Running...");
            mTextView.setText(" ");

            Thread thread = new Thread(MainActivity.this);
            thread.start();
        });

        try {
            torchscript_module = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "my_model_lite.ptl"));

        } catch (IOException e) {
            Log.e("Demo App", "Error reading assets", e);
            finish();
        }
    }


    @Override
    public void run() {
        // From Bitmap to OpenCV Mat
        Mat image_rgba = new Mat(mBitmap.getHeight(), mBitmap.getWidth(), CvType.CV_16UC3);
        Utils.bitmapToMat(mBitmap, image_rgba);

        // Remove Alpha channel and Convert it to Float
        Mat image_rgb = convertRGBAtoRGB(image_rgba);
        Mat image_float = new Mat();
        image_rgb.convertTo(image_float, CvType.CV_32F);

        // To FloatArray and Swap Axes
        float[][][] floatArray = convertMatToFloatArray(image_float);
        float[][][] transformedMatrix = transformMatrix(floatArray);

        // Convert to Tensor and Normalize
        Tensor inputTensor = convertToTensorAndNormalize(transformedMatrix);
        IValue output = torchscript_module.forward(IValue.from(inputTensor));
        float[] preds = output.toTensor().getDataAsFloatArray();

        // Getting the argmax of preds
        float max = preds[0];
        for (int i = 1; i < preds.length; i++) {
            if (preds[i] > max) {
                max = preds[i];
                argmax = i;
            }
        }

        runOnUiThread(() -> {
            mButton.setEnabled(true);
            mButton.setText("Recognize");
            mTextView.setText("Class Detected: "+ classes[argmax]);

        });
    }
    static Mat convertRGBAtoRGB(Mat image_rgba){
        Mat image_rgb = new Mat(image_rgba.rows(), image_rgba.cols(), CvType.CV_16UC3);
        List<Mat> channels = new ArrayList<>(3);
        Core.split(image_rgba, channels);
        List<Mat> targetChannels = channels.subList(0, 3); // Keep the first 3 channels (RGB)
        Core.merge(targetChannels, image_rgb);
        return image_rgb;
    }
    private static float[][][] transformMatrix(float[][][] originalMatrix) {
        int depth = originalMatrix.length;
        int rows = originalMatrix[0].length;
        int cols = originalMatrix[0][0].length;

        float[][][] transformedMatrix = new float[cols][depth][rows];

        for (int i = 0; i < originalMatrix.length; i++) {
            for (int j = 0; j < originalMatrix[i].length; j++) {
                // Access the values at floatArray[i][j] and assign them to newfloatArray
                float value1 = originalMatrix[i][j][0];
                float value2 = originalMatrix[i][j][1];
                float value3 = originalMatrix[i][j][2];

                // Assign values to the new array with swapped axes
                transformedMatrix[0][i][j] = value1;
                transformedMatrix[1][i][j] = value2;
                transformedMatrix[2][i][j] = value3;
            }
        }

        return transformedMatrix;
    }
    private static float[][][] convertMatToFloatArray(Mat originalMat) {
        int rows = originalMat.rows();
        int cols = originalMat.cols();
        int depth = originalMat.channels();

        float[][][] floatArray = new float[rows][cols][depth];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float[] pixel = new float[depth];
                originalMat.get(i, j, pixel);

                for (int k = 0; k < depth; k++) {
                    floatArray[i][j][k] = pixel[k];
                }
            }
        }

        return floatArray;
    }

    private static Tensor convertToTensorAndNormalize(float[][][] array) {
        int dim1 = array.length;
        int dim2 = array[0].length;
        int dim3 = array[0][0].length;

        // Flatten the array into a 1D float array
        float[] flatArray = new float[dim1 * dim2 * dim3];
        int index = 0;
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                for (int k = 0; k < dim3; k++) {
                    flatArray[index++] = array[i][j][k] / 255.0f;
                }
            }
        }

        // Create a PyTorch Tensor from the flat array and shape
        return Tensor.fromBlob(flatArray, new long[]{dim1, dim2, dim3});
    }



}