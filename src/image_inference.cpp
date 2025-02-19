/**
 * @file image_inference.cpp
 * @brief Object detection in a static image using YOLOv12 model.
 * 
 * This file implements an object detection application that utilizes YOLO 
 * (You Only Look Once) model, specifically version 12. 
 *
 * The application loads a specified image, processes it to detect objects, 
 * and displays the results with bounding boxes around detected objects.
 *
 * The application supports the following functionality:
 * - Loading a specified image from disk.
 * - Initializing the YOLO12 detector model and labels.
 * - Detecting objects within the image.
 * - Drawing bounding boxes around detected objects and displaying the result.
 * - Saving the processed image to a specified directory.
 *
 * Configuration parameters can be adjusted to suit specific requirements:
 * - `isGPU`: Set to true to enable GPU processing for improved performance; 
 *   set to false for CPU processing.
 * - `labelsPath`: Path to the class labels file (e.g., COCO dataset).
 * - `imagePath`: Path to the image file to be processed (e.g., dogs.jpg).
 * - `modelPath`: Path to the YOLO model file (e.g., ONNX format).
 * - `savePath`: Directory path to save the output image.
 *
 * Usage Instructions:
 * 1. Compile the application with the necessary OpenCV and YOLO dependencies.
 * 2. Ensure that the specified image and model files are present in the 
 *    provided paths.
 * 3. Run the executable to initiate the object detection process.
 *
 * Author: Mohamed Samir, https://www.linkedin.com/in/mohamed-samir-7a730b237/
 * Date: 19.02.2025
 */

// Include necessary headers
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <string>
#include "YOLO12.hpp"

int main() {
    // Paths to the model, labels, test image, and save directory
    const std::string labelsPath = "../models/coco.names";
    const std::string imagePath = "../data/dog.jpg";           // Image path
    const std::string savePath = "../data/dog_detections.jpg";   // Save directory

    // Model path for YOLOv12
    const std::string modelPath = "../models/yolov12n.onnx";   // YOLOv12

    // Initialize the YOLO detector with the chosen model and labels
    bool isGPU = true; // Set to false for CPU processing
    YOLO12Detector detector(modelPath, labelsPath, isGPU);

    // Load an image
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Could not open or find the image!\n";
        return -1;
    }

    // Detect objects in the image and measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Detection> results = detector.detect(image);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - start);

    std::cout << "Detection completed in: " << duration.count() << " ms" << std::endl;

    // Draw bounding boxes on the image
    detector.drawBoundingBox(image, results); // Simple bounding box drawing
    // detector.drawBoundingBoxMask(image, results); // Uncomment for mask drawing

    // Save the processed image to the specified directory
    if (cv::imwrite(savePath, image)) {
        std::cout << "Processed image saved successfully at: " << savePath << std::endl;
    } else {
        std::cerr << "Error: Could not save the processed image to: " << savePath << std::endl;
    }

    // Display the image
    cv::imshow("Detections", image);
    cv::waitKey(0); // Wait for a key press to close the window

    return 0;
}
