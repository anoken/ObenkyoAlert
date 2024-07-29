#include "nanodet_hand.h"
#include <alsa/asoundlib.h>
#include <dirent.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

// Global variables
ncnn::Net nanodet, mobilenet;
std::atomic<float> volumeLevel(0.2f);
std::atomic<double> currentRMS(0.0);
std::atomic<bool> isPlaying(false);
std::atomic<bool> running(true);
std::atomic<int> detec_class(-10);
bool debug_flg = 1;
const int nanodet_target_size = 320;
int mobilenet_target_size = 224 / 4;
float prob_threshold = 0.4f;
float nms_threshold = 0.2f;
cv::Mat overlay;
std::vector<std::vector<float>> m_weights;
std::vector<float> m_bias;
std::vector<std::vector<float>> pos_dat;
std::vector<std::vector<float>> neg_dat;

// Struct definitions
struct WAVHeader {
  char riff[4];
  int chunkSize;
  char wave[4];
  char fmt[4];
  int subchunk1Size;
  short audioFormat;
  short numChannels;
  int sampleRate;
  int byteRate;
  short blockAlign;
  short bitsPerSample;
  char data[4];
  int dataSize;
};

struct SharedFrame {
  cv::Mat frame;
  std::mutex mutex;
  std::condition_variable cond;
  std::atomic<bool> new_frame{false};
  std::atomic<bool> done{false};
};

// Function prototypes
void SEFR_init();
void SEFR_train(std::vector<std::vector<float>>& input_data, std::vector<int>& target_class);
uint8_t SEFR_predict(std::vector<float> new_image, std::vector<float>& score);
std::string getCurrentTimeStamp();


// Get current RSS (Resident Set Size) in MB
double getCurrentRSSInMB() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<double>(usage.ru_maxrss) / 1024.0;
    }
    return 0.0;
}

// Detect objects using NanoDet
static int detect_nanodet(const cv::Mat &bgr, std::vector<Object> &objects) {
  int width = bgr.cols;
  int height = bgr.rows;

  // pad to multiple of 32
  int w = width;
  int h = height;
  float scale = 1.f;
  if (w > h) {
    scale = (float)nanodet_target_size / w;
    w = nanodet_target_size;
    h = h * scale;
  } else {
    scale = (float)nanodet_target_size / h;
    h = nanodet_target_size;
    w = w * scale;
  }

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR,
                                               width, height, w, h);

  // pad to target_size rectangle
  int wpad = (w + 31) / 32 * 32 - w;
  int hpad = (h + 31) / 32 * 32 - h;
  ncnn::Mat in_pad;
  ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2,
                         wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

  const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
  const float norm_vals[3] = {0.017429f, 0.017507f, 0.017125f};
  in_pad.substract_mean_normalize(mean_vals, norm_vals);

  // cv::Mat in_debug = ncnnMatToCvMat(in_pad);
  // cv::imshow("detect_nanodet_debug", in_debug);

  ncnn::Extractor ex = nanodet.create_extractor();

  ex.input("input.1", in_pad);

  std::vector<Object> proposals;

  // stride 8
  {
    ncnn::Mat cls_pred;
    ncnn::Mat dis_pred;
    ex.extract("cls_pred_stride_8", cls_pred);
    ex.extract("dis_pred_stride_8", dis_pred);

    std::vector<Object> objects8;
    generate_proposals(cls_pred, dis_pred, 8, in_pad, prob_threshold, objects8);

    proposals.insert(proposals.end(), objects8.begin(), objects8.end());
  }

  // stride 16
  {
    ncnn::Mat cls_pred;
    ncnn::Mat dis_pred;
    ex.extract("cls_pred_stride_16", cls_pred);
    ex.extract("dis_pred_stride_16", dis_pred);

    std::vector<Object> objects16;
    generate_proposals(cls_pred, dis_pred, 16, in_pad, prob_threshold,
                       objects16);

    proposals.insert(proposals.end(), objects16.begin(), objects16.end());
  }

  // stride 32
  {
    ncnn::Mat cls_pred;
    ncnn::Mat dis_pred;
    ex.extract("cls_pred_stride_32", cls_pred);
    ex.extract("dis_pred_stride_32", dis_pred);

    std::vector<Object> objects32;
    generate_proposals(cls_pred, dis_pred, 32, in_pad, prob_threshold,
                       objects32);

    proposals.insert(proposals.end(), objects32.begin(), objects32.end());
  }

  // sort all proposals by score from highest to lowest
  qsort_descent_inplace(proposals);

  // apply nms with nms_threshold
  std::vector<int> picked;
  nms_sorted_bboxes(proposals, picked, nms_threshold);

  int count = picked.size();

  objects.resize(count);
  for (int i = 0; i < count; i++) {
    objects[i] = proposals[picked[i]];

    // adjust offset to original unpadded
    float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
    float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
    float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
    float y1 =
        (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

    // clip
    x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
    y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
    x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
    y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

    objects[i].rect.x = x0;
    objects[i].rect.y = y0;
    objects[i].rect.width = x1 - x0;
    objects[i].rect.height = y1 - y0;
  }

  return 0;
}

// Crop image based on detected objects
static void img_clop(cv::Mat &rgb, cv::Mat &cropped_image,
                     const std::vector<Object> &objects) {
  if (rgb.empty()) {
    return;
  }

  if (objects.size() > 0) {
    const Object &obj = objects[0];
    int img_x = obj.rect.x;
    int img_y = obj.rect.y;
    int img_width = obj.rect.width;
    int img_height = obj.rect.height;

    cv::Rect img_rect(img_x, img_y, img_width, img_height);

    cropped_image = safeRectCrop(rgb, img_rect);
  }
  return;
}

// Initialize hand class detection
static void hand_class_detec_init() {
  std::vector<cv::Mat> input_image_class1;
  std::vector<cv::Mat> input_image_class2;
  std::vector<cv::Mat> input_image_class3;
  std::vector<int> input_class;

  std::string folderPath1 = "../image/class1";
  std::string folderPath2 = "../image/class2";
  std::string folderPath3 = "../image/class3";

  folder_read(folderPath1, input_image_class1);
  folder_read(folderPath2, input_image_class2);
  folder_read(folderPath3, input_image_class3);

  std::cout << "input_image_class1=" << input_image_class1.size() << std::endl;
  std::cout << "input_image_class2=" << input_image_class2.size() << std::endl;
  std::cout << "input_image_class3=" << input_image_class3.size() << std::endl;

  std::vector<cv::Mat> image_merge;

  for (int i = 0; i < input_image_class1.size(); i++) {
    image_merge.push_back(input_image_class1[i]);
    input_class.push_back(0);
  }
  for (int i = 0; i < input_image_class2.size(); i++) {
    image_merge.push_back(input_image_class2[i]);
    input_class.push_back(1);
  }
  for (int i = 0; i < input_image_class3.size(); i++) {
    image_merge.push_back(input_image_class3[i]);
    input_class.push_back(2);
  }

  std::vector<std::vector<float>> input_merge;

  std::cout << "mobile predict:" << std::endl;

  for (unsigned int i = 0; i < image_merge.size(); i++) {
    ncnn::Mat in_data = ncnn::Mat::from_pixels_resize(
        image_merge[i].data, ncnn::Mat::PIXEL_RGB2BGR, image_merge[i].cols,
        image_merge[i].rows, mobilenet_target_size, mobilenet_target_size);

    const float norm_vals[3] = {1 / 255.0, 1 / 255.0, 1 / 255.0};
    in_data.substract_mean_normalize(0, norm_vals);
    if (debug_flg) {
      cv::Mat clop_in_debug = ncnnMatToCvMat(in_data);
      cv::imshow("clop_ncnn_img", clop_in_debug);
      char key = cv::waitKey(10);
    }
    ncnn::Mat out;

    const std::vector<const char *> &input_names = mobilenet.input_names();
    ncnn::Extractor ex = mobilenet.create_extractor();

    ex.input(input_names[0], in_data);  //
    ex.extract("31", out);              //

    std::vector<float> feature_n;

    for (int i = 0; i < out.w * out.h; i++) {
      feature_n.push_back(out[i]);
    }
    input_merge.push_back(feature_n);
  }
  std::cout << "SEFR_train:" << std::endl;

  SEFR_train(input_merge, input_class);

  std::cout << "SEFR_predict:" << std::endl;

  std::string filename = "SEFR_predict.txt";
  std::ofstream outFile(filename);
  if (!outFile.is_open()) {
    std::cerr << "Error: Unable to open file " << filename << " for writing."
              << std::endl;
    return;
  }
  for (int i = 0; i < image_merge.size(); i++) {
    std::vector<float> score;
    int label = SEFR_predict(input_merge[i], score);
    outFile << "No:" << i << " input_class:" << input_class[i]
            << ", label:" << label << "," << score[0] << "," << score[1] << ","
            << score[2] << "," << std::endl;

    std::cout << "No:" << i << " input_class:" << input_class[i]
              << ", label:" << label << "," << score[0] << "," << score[1]
              << "," << score[2] << "," << std::endl;
  }

  std::vector<cv::Mat> input_image_class_ex;
  std::string folderPath_ex =
      "../image/class_ex";  // 読み込む画像があるフォルダのパス

  folder_read(folderPath_ex, input_image_class_ex);
  std::vector<std::vector<float>> input_ex;

  for (unsigned int i = 0; i < input_image_class_ex.size(); i++) {
    ncnn::Mat in_data = ncnn::Mat::from_pixels_resize(
        input_image_class_ex[i].data, ncnn::Mat::PIXEL_RGB2BGR,
        input_image_class_ex[i].cols, input_image_class_ex[i].rows,
        mobilenet_target_size, mobilenet_target_size);

    const float norm_vals[3] = {1 / 255.0, 1 / 255.0, 1 / 255.0};
    in_data.substract_mean_normalize(0, norm_vals);

    ncnn::Mat out;

    const std::vector<const char *> &input_names = mobilenet.input_names();
    ncnn::Extractor ex = mobilenet.create_extractor();

    ex.input(input_names[0], in_data);  //
    ex.extract("31", out);              //

    std::vector<float> feature_n;

    for (int i = 0; i < out.w * out.h; i++) {
      feature_n.push_back(out[i]);
    }
    input_ex.push_back(feature_n);
  }
  outFile << "ex_file" << std::endl;
  for (int i = 0; i < input_ex.size(); i++) {
    std::vector<float> score;
    int label = SEFR_predict(input_ex[i], score);
    outFile << "No:" << i << " input_class:" << input_class[i]
            << ", label:" << label << "," << score[0] << "," << score[1] << ","
            << score[2] << "," << std::endl;

    std::cout << "No:" << i << " input_class:" << input_class[i]
              << ", label:" << label << "," << score[0] << "," << score[1]
              << "," << score[2] << "," << std::endl;
  }

  return;
}

// Detect hand class
static int hand_class_detec(cv::Mat &cropped_image) {
  int width = cropped_image.cols;
  int height = cropped_image.rows;

  ncnn::Mat in_data = ncnn::Mat::from_pixels_resize(
      cropped_image.data, ncnn::Mat::PIXEL_RGB2BGR, width, height,
      mobilenet_target_size, mobilenet_target_size);

  const float norm_vals[3] = {1 / 255.0, 1 / 255.0, 1 / 255.0};
  in_data.substract_mean_normalize(0, norm_vals);
  if (debug_flg) {
    cv::Mat clop_in_debug = ncnnMatToCvMat(in_data);
    cv::imshow("clop_ncnn_img", clop_in_debug);
  }
  ncnn::Mat out;

  const std::vector<const char *> &input_names = mobilenet.input_names();

  ncnn::Extractor ex = mobilenet.create_extractor();

  ex.input(input_names[0], in_data);
  ex.extract("31", out);

  static std::vector<float> feature_n;
  static std::vector<float> feature_ave;

  static std::vector<std::vector<float>> feature_matrix;
  static std::vector<int> class_matrix;

  if (feature_n.size() == 0) {
    feature_n.resize(out.w * out.h);
    feature_ave.resize(feature_n.size());
  }

  float feature_ww = 0.5;

  for (int i = 0; i < out.w * out.h; i++) {
    feature_n[i] = out[i];
  }
  for (int i = 0; i < out.w * out.h; i++) {
    feature_ave[i] =
        feature_ww * feature_ave[i] + (1.0 - feature_ww) * feature_n[i];
  }
  std::vector<float> score;

  int label = SEFR_predict(feature_n, score);

  if (debug_flg) {
    std::cout << "label:" << label << "," << score[0] << "," << score[1] << ","
              << score[2] << "," << std::endl;
    cv::Mat resultMap;
    ncnn_conv_cvMat(feature_ave, resultMap);
    if (debug_flg) cv::imshow("feature_ave", resultMap);
    ncnn_conv_cvMat(feature_n, resultMap);
    if (debug_flg) cv::imshow("feature_n", resultMap);
  }

  return label;
}

// Draw detected objects on the image
static void draw_objects(cv::Mat &rgb, const std::vector<Object> &objects,
                         int label = -1) {
  for (size_t i = 0; i < objects.size(); i++) {
    const Object &obj = objects[i];

    if (obj.rect.width == obj.rect.height)  // left_hand_0_or_right_hand_1
      cv::rectangle(rgb, obj.rect, cv::Scalar(255, 0, 0));
    else
      cv::rectangle(rgb, obj.rect, cv::Scalar(0, 0, 255));

    if (label >= 0) {
      std::string message;

      if (label == 0)
        message = "Studying";
      else if (label == 1)
        message = "Not Studying";
      else if (label == 2)
        message = "SmartPhone";

      int font_face = cv::FONT_HERSHEY_SIMPLEX;
      double font_scale = 0.5;
      int thickness = 1;
      int baseline = 0;

      cv::Size text_size =
          cv::getTextSize(message, font_face, font_scale, thickness, &baseline);
      cv::Point text_org(obj.rect.x, obj.rect.y - text_size.height - 5);
      cv::Scalar color2(255, 115, 55);
      cv::putText(rgb, message, text_org, font_face, font_scale, color2,
                  thickness);
    }
  }
}

// Callback for probability threshold trackbar
void onTrackbar_prob(int value, void *userdata) {
  // Process when the parameter is changed
  prob_threshold = static_cast<float>(value) / 100.0;
  std::cout << "Probability threshold: " << prob_threshold << std::endl;
}

// Callback for NMS threshold trackbar
void onTrackbar_nms(int value, void *userdata) {
  // Process when the parameter is changed
  nms_threshold = static_cast<float>(value) / 100.0;
  std::cout << "NMS threshold: " << nms_threshold << std::endl;
}

// Signal handler for graceful shutdown
void signalHandler(int signum) {
  std::cout << "Interrupt signal (" << signum << ") received.\n";
  running = false;
}

// Draw face on image based on volume
void draw_face(cv::Mat &image, float volume) {
  cv::Scalar front_col(0, 0, 0);     // Black
  cv::Scalar back_col(255, 128, 0);  // Initial Yellow (BGR format)

  image = cv::Scalar(back_col[0], back_col[1], back_col[2]);  // Fill background

  // 左上の64x64ピクセルの領域にoverlayを張り付ける
  if (!overlay.empty()) {
    // cv::Mat roi = image(cv::Rect(0, 0, 64, ));
    // cv::Mat resized_overlay;
    // cv::resize(overlay, resized_overlay, cv::Size(64, 64));
    // resized_overlay.copyTo(roi);
    const float scale = 1.5;

    int bottom = image.rows - 64 * scale;  // 下端から64ピクセル上
    cv::Mat roi = image(cv::Rect(0, bottom, 64 * scale, 64 * scale));
    cv::Mat resized_overlay;
    cv::resize(overlay, resized_overlay, cv::Size(64 * scale, 64 * scale));
    resized_overlay.copyTo(roi);
  }

  // Left eyebrow
  std::vector<cv::Point> left_eyebrow = {
      {190, 70}, {190, 50}, {280 + random(10), 25 + random(15)}};
  cv::fillConvexPoly(image, left_eyebrow, front_col);
  // Right eyebrow
  std::vector<cv::Point> right_eyebrow = {
      {130, 70}, {130, 50}, {50 + random(10), 25 + random(15)}};
  cv::fillConvexPoly(image, right_eyebrow, front_col);
  // Left eye
  cv::circle(image, cv::Point(90 + random(5), 93 + random(5)), 25, front_col,
             -1);
  // Right eye
  cv::circle(image, cv::Point(230 + random(5), 93 + random(5)), 25, front_col,
             -1);

  float max_open = 60.0f;  // 最大の口の開き
  float open = std::min(volume * max_open, max_open);

  // Mouth
  std::vector<cv::Point> mouth = {{133, static_cast<int>(188 - open / 2)},
                                  {193, static_cast<int>(188 - open / 2)},
                                  {193, static_cast<int>(188 + open / 2)},
                                  {133, static_cast<int>(188 + open / 2)}};
  cv::fillConvexPoly(image, mouth, front_col);
}

// Display frame buffer information
void view_information(const fb_fix_screeninfo &finfo,
                      const fb_var_screeninfo &vinfo) {
  // フレームバッファの情報を表示する関数の実装
  std::cout << "Fixed screen info:" << std::endl;
  std::cout << "  smem_len: " << finfo.smem_len << std::endl;
  std::cout << "Variable screen info:" << std::endl;
  std::cout << "  xres: " << vinfo.xres << std::endl;
  std::cout << "  yres: " << vinfo.yres << std::endl;
  std::cout << "  bits_per_pixel: " << vinfo.bits_per_pixel << std::endl;
}



// Play WAV file
void playWAV(const std::string &filename) {
  isPlaying.store(true);

  std::ifstream wavFile(filename, std::ios::binary);
  if (!wavFile) {
    std::cerr << "Error opening WAV file." << std::endl;
    return;
  }
  std::cout << "wavFile readed" << std::endl;

  WAVHeader header;
  wavFile.read(reinterpret_cast<char *>(&header), sizeof(WAVHeader));

  int samplesPerInterval = header.sampleRate / 30;
  int bytesPerSample = header.bitsPerSample / 8;
  int bytesPerInterval =
      samplesPerInterval * header.numChannels * bytesPerSample;

  snd_pcm_t *handle;
  std::cout << "snd_pcm_open" << std::endl;

  if (snd_pcm_open(&handle, "default", SND_PCM_STREAM_PLAYBACK, 0) < 0) {
    std::cerr << "Error opening PCM device." << std::endl;
    return;
  }
  std::cout << "params set" << std::endl;

  snd_pcm_hw_params_t *params;
  snd_pcm_hw_params_alloca(&params);
  snd_pcm_hw_params_any(handle, params);
  snd_pcm_hw_params_set_access(handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);
  snd_pcm_hw_params_set_format(handle, params, SND_PCM_FORMAT_S16_LE);
  snd_pcm_hw_params_set_channels(handle, params, header.numChannels);
  snd_pcm_hw_params_set_rate_near(
      handle, params, reinterpret_cast<unsigned int *>(&header.sampleRate), 0);
  snd_pcm_hw_params_set_buffer_size(handle, params, samplesPerInterval * 2);
  snd_pcm_hw_params(handle, params);

  std::vector<int16_t> buffer(samplesPerInterval * header.numChannels);
  std::cout << "playWAV inited" << std::endl;

  while (
      wavFile.read(reinterpret_cast<char *>(buffer.data()), bytesPerInterval) &&
      isPlaying.load()) {
    for (auto &sample : buffer) {
      sample = static_cast<int16_t>(sample * volumeLevel.load());
    }

    snd_pcm_writei(handle, buffer.data(), samplesPerInterval);
    currentRMS.store(calculateRMS(buffer));
  }

  isPlaying.store(false);
  snd_pcm_drain(handle);
  snd_pcm_close(handle);
}

// Thread function for console input
void consoleInputThread() {
  std::string input;
  while (running) {
    std::getline(std::cin, input);
    if (input == "q") {
      running.store(false);
      break;
    } else if (input.substr(0, 3) == "vol") {
      try {
        float newVolume = std::stof(input.substr(3));
        if (newVolume >= 0.0f && newVolume <= 1.0f) {
          volumeLevel.store(newVolume);
          std::cout << "Volume set to " << newVolume << std::endl;
        } else {
          std::cout << "Invalid volume level. Use a value between 0.0 and 1.0"
                    << std::endl;
        }
      } catch (...) {
        std::cout << "Invalid volume command. Use 'vol' followed by a number "
                     "between 0.0 and 1.0"
                  << std::endl;
      }
    }
  }
}

// Thread function for speech output
void spreakThread() {
  int label_old = -10;
  int label_counter = 0;
  const int label_counter_thresh = 3;

  while (running) {
    int label = detec_class.load();
    std::cout << "label:" << label << "\n";
    if ((label == 0) && (label_old == 0)) {
      //  label_counter++;
    } else if ((label == 1) && (label_old == 1)) {
      label_counter++;
    } else if ((label == 2) && (label_old == 2)) {
      label_counter++;
    } else {
      label_counter = 0;
    }

    if (label_counter > label_counter_thresh) {
      label_counter = 0;

      if (label == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      } else if (label == 1) {
        std::thread playWAVThread(playWAV, "../image/benkyo.wav");
        playWAVThread.join();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

      } else if (label == 2) {
        std::thread playWAVThread(playWAV, "../image/sumaho.wav");
        playWAVThread.join();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }
    }

    label_old = label;
     std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  return;
}

// Thread function for NCNN processing
void ncnnprocThread(SharedFrame &shared) {
  ncnn::set_cpu_powersave(2);
  nanodet.opt = ncnn::Option();
  nanodet.opt.num_threads = ncnn::get_big_cpu_count();

  nanodet.load_param("../models/nanodet-hand.param");
  nanodet.load_model("../models/nanodet-hand.bin");

  const std::vector<const char *> &input_names = nanodet.input_names();
  const std::vector<const char *> &output_names = nanodet.output_names();

  for (size_t i = 0; i < input_names.size(); i++) {
    std::cout << "input_names:" << input_names[i] << std::endl;
  }
  for (size_t i = 0; i < output_names.size(); i++) {
    std::cout << "output_names:" << output_names[i] << std::endl;
  }

  mobilenet.load_param("../models/mobilenet1.0.ncnn.param");
  mobilenet.load_model("../models/mobilenet1.0.ncnn.bin");

  const std::vector<const char *> &input_names2 = mobilenet.input_names();
  const std::vector<const char *> &output_names2 = mobilenet.output_names();

  for (size_t i = 0; i < input_names2.size(); i++) {
    std::cout << "input_names:" << input_names2[i] << std::endl;
  }
  for (size_t i = 0; i < output_names2.size(); i++) {
    std::cout << "output_names:" << output_names2[i] << std::endl;
  }

  SEFR_init();
  hand_class_detec_init();

  std::chrono::steady_clock::time_point Tbegin, Tend1, Tend2;

  while (running) {
    std::unique_lock<std::mutex> lock(shared.mutex);
    shared.cond.wait(lock, [&] { return shared.new_frame || shared.done; });

    if (shared.new_frame) {
      cv::Mat clop_img;

      cv::Mat frame = shared.frame.clone();
      shared.new_frame = false;
      lock.unlock();

      Tbegin = std::chrono::steady_clock::now();

      detect_nanodet(frame, obj);
      obj_norm.resize(obj.size());
      Tend1 = std::chrono::steady_clock::now();

     // Transform the image area extracted by nanodet into a square based on the longer side.
      for (int i = 0; i < obj.size(); i++) {
        float cent_x = obj[i].rect.x + obj[i].rect.width / 2;
        float cent_y = obj[i].rect.y + obj[i].rect.height / 2;
        float w = obj[i].rect.width;
        float h = obj[i].rect.height;

        if (w >= h) {
          obj_norm[i].rect.width = w;
          obj_norm[i].rect.height = w;
          obj_norm[i].rect.x = cent_x - w / 2;
          obj_norm[i].rect.y = cent_y - w / 2;

        } else {
          obj_norm[i].rect.width = h;
          obj_norm[i].rect.height = h;
          obj_norm[i].rect.x = cent_x - h / 2;
          obj_norm[i].rect.y = cent_y - h / 2;
        }
      }

      // Reasoning with mobilenet+SEFR
            int label = -1;
      if (obj.size() > 0) {
        img_clop(frame, clop_img, obj_norm);
        label = hand_class_detec(clop_img);
        detec_class.store(label);
        // overlay = clop_img.clone();
      }
      Tend2 = std::chrono::steady_clock::now();

      nanodet_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(Tend1 - Tbegin)
              .count();
      mobilenet_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(Tend2 - Tend1)
              .count();

    } else if (shared.done) {
      break;
    }
  }
}

bool saveImageToFolder(const cv::Mat &image, const std::string &folderName,
                       const std::string &prefix,
                       const std::string &timestamp) {
  std::string folderPath = folderName;

  // Create folder if it does not exist
    struct stat st = {0};
  if (stat(folderPath.c_str(), &st) == -1) {
    if (mkdir(folderPath.c_str(), 0700) == -1) {
      std::cerr << "Error creating directory: " << strerror(errno) << std::endl;
      return false;
    }
  }

  std::string filename = prefix + "_" + timestamp + ".png";
  std::string filePath = folderPath + "/" + filename;

// Save Image
  try {
    cv::imwrite(filePath, image);
    std::cout << "Image saved: " << filePath << std::endl;
    return true;
  } catch (const cv::Exception &e) {
    std::cerr << "Error saving image: " << e.what() << std::endl;
    return false;
  }
}

int main(int argc, char **argv) {

// Set signal handler
  signal(SIGINT, signalHandler);
  std::cout << "start" << std::endl;
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "memory used: " << getCurrentRSSInMB() << " MB"
            << std::endl;
  cv::VideoCapture cap;
#if defined(__x86_64__)
  int max_value = 100;
  int prob_threshold_int = prob_threshold * 100;
  int nms_threshold_int = nms_threshold * 100;
  cv::namedWindow("frame", cv::WINDOW_AUTOSIZE);
  // トラックバーを作成
  cv::createTrackbar("prob_th", "frame", &prob_threshold_int, max_value,
                     onTrackbar_prob);
  cv::createTrackbar("nms_th", "frame", &nms_threshold_int, max_value,
                     onTrackbar_nms);

  //  cap.open("/dev/video0");
  cap.open(0);
  if (!cap.set(cv::CAP_PROP_FRAME_WIDTH, 640))
    std::cout << "camera set CAP_PROP_FRAME_WIDTH error" << std::endl;
  if (!cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480))
    std::cout << "camera set CAP_PROP_FRAME_HEIGHT error" << std::endl;
  if (!cap.set(cv::CAP_PROP_FPS, 30))
    std::cout << "camera set CAP_PROP_FPS error" << std::endl;
  cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  debug_flg = 1;
  cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

#endif

#if defined(__arm__)
  cap.open(0);
  if (!cap.set(cv::CAP_PROP_FRAME_WIDTH, 640))
    std::cout << "camera set CAP_PROP_FRAME_WIDTH error" << std::endl;
  if (!cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480))
    std::cout << "camera set CAP_PROP_FRAME_HEIGHT error" << std::endl;
  debug_flg = 0;
#endif

  SharedFrame shared;
  std::thread displayThread(displayFace_thread);
  std::thread inputThread(consoleInputThread);
  std::thread ncnnThread(ncnnprocThread, std::ref(shared));
  std::thread speakThread(spreakThread);

  std::cout << "Start grabbing, press ESC on Live window to terminate"
            << std::endl;
  while (running) {
    cv::Mat frame, clop_img;

    cap >> frame;

    if (frame.empty()) {
      std::cerr << "ERROR: Unable to grab from the camera" << std::endl;
      break;
    }
    // 切り取った画像を320x240にリサイズする
    cv::resize(frame, frame, cv::Size(320, 240));

    std::unique_lock<std::mutex> lock(shared.mutex);
    shared.frame = frame.clone();
    shared.new_frame = true;
    lock.unlock();
    shared.cond.notify_one();
    int label = detec_class.load();

    img_clop(frame, clop_img, obj_norm);

    draw_objects(frame, obj);
    draw_objects(frame, obj_norm, label);

    putText(frame, cv::format("label=%d", label), cv::Point(10, 10),
            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255));


    // メモリ使用量を取得して文字列に変換
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "memory: " << getCurrentRSSInMB() << " MB";
    std::string memoryText = oss.str();

    cv::putText(frame, memoryText,
                cv::Point(10, frame.rows - 10),  // 左下隅に近い位置
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,  // フォントとスケール（小さめ）
                cv::Scalar(0, 255, 255), 1);  // 黄色、線の太さ

    overlay = frame.clone();

    static int capture_num = 0;

    if (cv_save_flg > 0) {
      std::cout << "capture_num" << capture_num << "cv_save_flg" << cv_save_flg
                << std::endl;

      if ((!clop_img.empty()) && (capture_num < cv_save_flg)) {
        std::string currentTime = getCurrentTimeStamp();
        bool success = saveImageToFolder(clop_img, "save", "hand", currentTime);
        if (success) {
          std::cout << "Image saved successfully." << std::endl;
        } else {
          std::cout << "Failed to save image." << std::endl;
        }
        capture_num++;
      }

      if (capture_num >= cv_save_flg) {
        cv_save_flg = 0;
        capture_num = 0;
      }
    }

    if (debug_flg) {
      imshow("frame", frame);
      char key = cv::waitKey(5);
      if (key == 27) break;

      if (key == 's' || key == 'S') {
        if (!clop_img.empty()) {
          std::string currentTime = getCurrentTimeStamp();
          std::string filename = "hand_" + currentTime + ".png";
          cv::imwrite(filename, clop_img);
          std::cout << "save: " << filename << std::endl;
        }
      }
    }

    std::this_thread::sleep_for(
        std::chrono::milliseconds(10));  // Changed to 10ms
  }

  std::cout << "Closing the camera" << std::endl;
  std::cout << "Main thread is stopping. Waiting for worker threads...\n";

  // すべてのスレッドが終了するのを待つ
  displayThread.join();
  inputThread.join();

  std::cout << "All threads have stopped. Exiting.\n";
  return 0;
}

void save_pos_dat_to_file(const std::vector<std::vector<float>> &pos_dat,
                          const std::string &base_filename) {
  std::string filename = base_filename + "_class_" + ".csv";
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filename << std::endl;
    return;
  }

  for (size_t c = 0; c < pos_dat.size(); ++c) {
    file << std::fixed
         << std::setprecision(6);  // 6桁の精度で浮動小数点数を出力

    for (const auto &value : pos_dat[c]) {
      file << value << ",";
    }
    file << "\n";
  }
  file.close();
  std::cout << "Saved data for class " << " to " << filename << std::endl;
}

void save_pos_dat_to_file(const std::vector<float> &pos_dat,
                          const std::string &base_filename) {
  std::string filename = base_filename + "_class_" + ".csv";
  std::ofstream file(filename);

  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filename << std::endl;
    return;
  }

  file << std::fixed << std::setprecision(6);  // 6桁の精度で浮動小数点数を出力

  for (const auto &value : pos_dat) {
    file << value << ",";
  }
  file << "\n";
  file.close();
  std::cout << "Saved data for class " << " to " << filename << std::endl;
}

void SEFR_init() {
  unsigned int class_num = 3;
  unsigned int data_size = 1024;
  m_weights.resize(class_num, std::vector<float>(data_size, 0.0));
  m_bias.resize(class_num, 0);
  pos_dat.resize(class_num, std::vector<float>(data_size, 0.0));
  neg_dat.resize(class_num, std::vector<float>(data_size, 0.0));
}

void SEFR_train(std::vector<std::vector<float>> &input_data,
                std::vector<int> &target_class) {
  std::cout << "input_data size" << input_data.size() << std::endl;

  unsigned int class_num = 3;

  // for (unsigned int c = 0; c < target_class.size(); c++) {
  //   std::cout << "target_class: " << target_class[c] << std::endl;
  // }

  // 行数（外側のベクトルのサイズ）を確認
  size_t input_size = input_data.size();

  // 列数（最初の行のサイズ）を確認
  size_t data_size = 0;
  if (!input_data.empty()) {
    data_size = input_data[0].size();
  }
  std::cout << "input_size: " << input_size << std::endl;
  std::cout << "data_size: " << data_size << std::endl;

  save_pos_dat_to_file(input_data, "input_data");

  std::cout << "pos_dat.size(): " << pos_dat.size() << std::endl;
  std::cout << "neg_dat.size(): " << neg_dat.size() << std::endl;
  std::cout << " input_data.size(): " << input_data.size() << std::endl;
  std::cout << " target_class.size(): " << target_class.size() << std::endl;

  for (unsigned int c = 0; c < class_num; c++) {
    for (unsigned int x = 0; x < data_size; x++) {
      float pos_cnt = 0;
      float nega_cnt = 0;
      for (unsigned int i = 0; i < input_size; i++) {
        if (target_class[i] != c) {
          pos_cnt++;
          pos_dat[c][x] = 1.0 / (pos_cnt + 1.0) *
                          (pos_cnt * pos_dat[c][x] + input_data[i][x]);
        } else {
          nega_cnt++;
          neg_dat[c][x] = 1.0 / (nega_cnt + 1.0) *
                          (nega_cnt * neg_dat[c][x] + input_data[i][x]);
        }
      }
    }
  }

  save_pos_dat_to_file(input_data, "input_data");
  save_pos_dat_to_file(pos_dat, "pos_dat");
  save_pos_dat_to_file(neg_dat, "neg_dat");

  //    ncnn_conv_cvMat(feature_n, resultMap);
  //    cv::imshow("feature_n", resultMap);

  for (unsigned int c = 0; c < class_num; c++) {
    for (unsigned int x = 0; x < data_size; x++) {
      float avg_pos = pos_dat[c][x];
      float avg_neg = neg_dat[c][x];
      float weights = (avg_pos - avg_neg) / (avg_pos + avg_neg + 1.0E-4);
      m_weights[c][x] = weights;
    }
  }

  save_pos_dat_to_file(m_weights, "m_weights");

  for (unsigned int c = 0; c < class_num; c++) {
    double avg_pos_w = 0.0, avg_neg_w = 0.0;
    unsigned int count_pos = 0, count_neg = 0;
    m_bias[c] = 0;
    for (unsigned int i = 0; i < input_size; i++) {
      double weighted_score = 0.0;
      for (unsigned int x = 0; x < data_size; x++) {
        double score = m_weights[c][x] * input_data[i][x];
        weighted_score += score;
      }
      if (target_class[i] != c) {
        avg_pos_w += weighted_score;
        count_pos++;
      } else {
        avg_neg_w += weighted_score;
        count_neg++;
      }
    }
    avg_pos_w /= double(count_pos);
    avg_neg_w /= double(count_neg);

    m_bias[c] =
        -1.0 * (double(count_neg) * avg_pos_w + double(count_pos) * avg_neg_w) /
        double(count_pos + count_neg + 0.0001);
  }

  save_pos_dat_to_file(m_bias, "m_bias");
}

uint8_t SEFR_predict(std::vector<float> new_image, std::vector<float> &score) {
  int class_num = 3;
  score.resize(class_num);

  if (m_weights.empty()) {
    std::cout << "m_weights.empty" << std::endl;
    return -1;
  }

  for (int c = 0; c < class_num; c++) {
    score[c] = 0.0;
    for (unsigned int x = 0; x < new_image.size(); x++) {
      // calculate weight of each labels
      score[c] += (new_image[x] * m_weights[c][x]);
    }
    score[c] += m_bias[c];  // add bias of each labels
  }

  float min_value = *std::min_element(score.begin(), score.end());
  auto min_index = std::distance(score.begin(),
                                 std::min_element(score.begin(), score.end()));

  return min_index;  // return prediction
}

int folder_read(std::string folderPath, std::vector<cv::Mat> &input_image) {
  std::cout << "file read" << folderPath << std::endl;

  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir(folderPath.c_str())) != NULL) {
    // ディレクトリのすべてのファイルをループで読み込む
    while ((ent = readdir(dir)) != NULL) {
      std::string fileName = ent->d_name;
      // JPEGまたはPNG画像のみを対象とする
      if (hasSuffix(fileName, ".jpg") || hasSuffix(fileName, ".png")) {
        std::string fullPath = folderPath + "/" + fileName;
        cv::Mat image = cv::imread(fullPath, cv::IMREAD_COLOR);
        if (!image.empty()) {
          std::cout << "image read success: " << fileName << std::endl;
          // ここで画像に対する処理を行う
          input_image.push_back(image);
        } else {
          std::cerr << "image read fail: " << fileName << std::endl;
        }
      }
    }
    closedir(dir);
  } else {
    std::cerr << "ディレクトリを開けませんでした: " << folderPath << std::endl;
    return EXIT_FAILURE;
  }

  return 0;
}