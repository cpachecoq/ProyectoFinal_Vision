#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/inference_session.h>
#include <onnxruntime/core/graph/graph.h>
#include <jni.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <iostream>
#include <android/log.h>
#include <opencv2/ml.hpp>
#include <android/bitmap.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

#define LOG_TAG "OpenCV_Native"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace std;
using namespace cv;
using namespace cv::ml;

std::string ModelPath;
jobject g_mainActivity = nullptr;

extern "C" JNIEXPORT void JNICALL
Java_ec_edu_ups_prueba2_MainActivity_findFeatures(JNIEnv* env, jobject /* this */, jlong matAddr) {
    Mat* mat = (Mat*)matAddr;

    // Verificar que la imagen sea en escala de grises
    if (mat->channels() != 1) {
        LOGE("La imagen debe ser en escala de grises. Convertiendo...");
        cvtColor(*mat, *mat, COLOR_BGR2GRAY); // Convertir a escala de grises si es necesario
    }

    // Redimensionar la imagen a 28x28 píxeles
    Mat resizedImage;
    resize(*mat, resizedImage, Size(28, 28)); // Tamaño debe ser 28x28 píxeles

    // Verificar dimensiones de la imagen redimensionada
    LOGI("Dimensiones de la imagen redimensionada: %d x %d", resizedImage.cols, resizedImage.rows);

    Size winSize(28, 28);

    Size cellSize(4, 4);

    Size blockSize(8, 8); // Ajusta esto si es necesario

    Size blockStride(4, 4); // Ajusta esto si es necesario

    int nbins = 9;

// Definir el descriptor HOG
    HOGDescriptor hog(winSize, blockSize, blockStride, cellSize, nbins);

    // Calcular las características HOG
    vector<float> hogFeatures;
    hog.compute(resizedImage, hogFeatures, Size(0, 0), Size(0, 0));

    // Verificar tamaño del vector de características
    size_t expectedSize = 2025; // Tamaño esperado basado en tus parámetros de entrenamiento
    LOGI("Número de características HOG calculadas: %zu", hogFeatures.size());

    if (hogFeatures.size() != expectedSize) {
        LOGE("Error: Tamaño de características HOG no coincide. Esperado: %zu, Obtido: %zu", expectedSize, hogFeatures.size());
        return;
    }

    // Log de las características HOG para depuración
    LOGI("Características HOG calculadas. Tamaño: %zu", hogFeatures.size());
    for (size_t i = 0; i < hogFeatures.size(); i++) {
        LOGI("Característica %zu: %f", i, hogFeatures[i]);
    }

    // Realizar la predicción con el modelo ONNX
    std::vector<int64_t> inputShape = {1, hogFeatures.size()};
    onnxruntime::TensorShape inputShapeVec(inputShape);
    auto inputTensor = onnxruntime::Ort::GetApi().CreateTensor<float>(hogFeatures.data(), hogFeatures.size());

    std::vector<onnxruntime::OrtValue> inputs;
    inputs.push_back(std::move(inputTensor));

    std::vector<onnxruntime::OrtValue> outputs;

// Obtener el valor de la predicción
    float* outputData = outputs[0].GetTensorMutableData<float>();
    float prediction = outputData[0]; // Suponiendo que es una predicción de clase única
    LOGI("Predicción del modelo: %f", prediction);

    if (g_mainActivity != nullptr) {
        jclass clazz = env->GetObjectClass(g_mainActivity);
        if (clazz == nullptr) {
            LOGE("No se pudo encontrar la clase MainActivity");
            return;
        }
        jmethodID methodID = env->GetMethodID(clazz, "updatePrediction", "(F)V");
        if (methodID == nullptr) {
            LOGE("No se pudo encontrar el método updatePrediction");
            return;
        }
        env->CallVoidMethod(g_mainActivity, methodID, prediction);
    } else {
        LOGE("Instancia de MainActivity es nula.");
    }

}

extern "C" JNIEXPORT void JNICALL
Java_ec_edu_ups_prueba2_MainActivity_setMainActivityInstance(JNIEnv* env, jobject /* this */, jobject mainActivityInstance) {
    // Guarda una referencia global a la instancia de MainActivity
    g_mainActivity = env->NewGlobalRef(mainActivityInstance);
}


extern "C" JNIEXPORT void JNICALL
Java_ec_edu_ups_prueba2_MainActivity_initModelPath(
        JNIEnv* env,
        jobject /* this */,
        jstring modelPath) {
    const char *modelPathChars = env->GetStringUTFChars(modelPath, 0);
    ModelPath = std::string(modelPathChars);
    env->ReleaseStringUTFChars(modelPath, modelPathChars);
    LOGI("Ruta del modelo: %s", ModelPath.c_str());

    std::ifstream file(ModelPath);
    if (file.good()) {
        LOGI("El archivo %s existe.", ModelPath.c_str());
    } else {
        LOGE("El archivo %s no existe.", ModelPath.c_str());
    }

    onnxruntime::Env env(onnxruntime::LoggingLevel::WARNING, "ONNXRuntime");
    onnxruntime::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(onnxruntime::GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    std::unique_ptr<onnxruntime::InferenceSession> session(new onnxruntime::InferenceSession(env, session_options));
    auto status = session->Load(ModelPath);
    if (!status.IsOK()) {
        LOGE("Error al cargar el modelo: %s", status.ErrorMessage().c_str());
        return;
    }

    status = session->Initialize();
    if (!status.IsOK()) {
        LOGE("Error al inicializar la sesión: %s", status.ErrorMessage().c_str());
        return;
    }

}




