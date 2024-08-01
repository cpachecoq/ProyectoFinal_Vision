package ec.edu.ups.prueba2;

import static android.content.ContentValues.TAG;

import org.opencv.core.Size;
import android.Manifest;
import android.annotation.SuppressLint;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.SurfaceHolder;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.Utils;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.IOException;
import java.io.File;
import java.io.FileOutputStream;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_PERMISSIONS = 100;
    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private ImageView imageView;
    private TextView resultTextView;
    private Button btnCapture;
    private Uri photoUri;

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "OpenCV initialization failed");
        } else {
            Log.d("OpenCV", "OpenCV initialization succeeded");
        }
        System.loadLibrary("prueba2"); // Cargar la biblioteca nativa
    }

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        resultTextView = findViewById(R.id.resultTextView);
        btnCapture = findViewById(R.id.btnCapture);

        copyAsset("modelo_hog_proyectoFinal.onnx");

        String modelPath = getFilesDir().getAbsolutePath() + "/modelo_hog_proyectoFinal.onnx";
        initModelPath(modelPath);

        btnCapture.setOnClickListener(v -> {
            if (allPermissionsGranted()) {
                openCamera();
            } else {
                requestPermissions();
            }
        });

    }

    private void openCamera() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            // Crear un archivo para guardar la imagen
            ContentValues values = new ContentValues();
            values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
            photoUri = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

            // Pasar el URI al Intent para que la cámara pueda guardar la foto en el archivo
            takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri);
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            try {
                // Obtener la imagen capturada
                InputStream imageStream = getContentResolver().openInputStream(photoUri);
                Bitmap bitmap = BitmapFactory.decodeStream(imageStream);

                // Convertir Bitmap a Mat
                Mat src = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC1);
                Utils.bitmapToMat(bitmap, src);

                // Convertir a escala de grises
                Mat gray = new Mat();
                Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);

                // Mostrar la imagen en el ImageView (opcional, si deseas mostrar la imagen en escala de grises)
                Bitmap grayBitmap = Bitmap.createBitmap(gray.cols(), gray.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(gray, grayBitmap);
                imageView.setImageBitmap(grayBitmap);

                // Aplicar procesamiento de la imagen
                findFeatures(gray.getNativeObjAddr());

                // Mostrar los resultado
                resultTextView.setVisibility(TextView.VISIBLE);
                imageView.setVisibility(ImageView.VISIBLE);
                //btnCapture.setVisibility(Button.GONE);

            } catch (FileNotFoundException e) {
                Log.e("FileError", "Error al cargar la imagen: " + e.getMessage());
            }
        }
    }

    // Declarar el método nativo
    public native void findFeatures(long matAddr);

    private boolean allPermissionsGranted() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestPermissions() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_PERMISSIONS);
    }

    public void updatePrediction(float prediction) {
        resultTextView.setText("Predicción: " + prediction);
    }


    private void copyAsset(String filename) {
        InputStream in = null;
        OutputStream out = null;
        try {
            in = getAssets().open(filename);
            String outFileName = getFilesDir().getAbsolutePath() + "/" + filename;
            out = new FileOutputStream(outFileName);
            copyFile(in, out);
            in.close();
            out.flush();
            out.close();


            File file = new File(outFileName);
            if (file.exists()) {
                Log.d(TAG, "El archivo " + filename + " fue copiado exitosamente a " + outFileName);
            } else {
                Log.e(TAG, "Error al copiar el archivo " + filename);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public native void initModelPath(String modelPath);

    private void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[1024];
        int read;
        while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSIONS) {
            if (allPermissionsGranted()) {
                openCamera();
            } else {
                Toast.makeText(this, "Permiso de cámara denegado", Toast.LENGTH_SHORT).show();
            }
        }
    }
}
