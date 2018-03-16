/*!
 *  Copyright (c) 2015 by Xiao Liu, pertusa, caprice-j
 * \file image_classification-predict.cpp
 * \brief C++ predict example of mxnet
 */

 //
 //  File: image-classification-predict.cpp
 //  This is a simple predictor which shows
 //  how to use c api for image classfication
 //  It uses opencv for image reading
 //  Created by liuxiao on 12/9/15.
 //  Thanks to : pertusa, caprice-j, sofiawu, tqchen, piiswrong
 //  Home Page: www.liuxiao.org
 //  E-mail: liuxiao@foxmail.com
 //

#define PREDICT_TEST
#ifdef PREDICT_TEST

// Path for c_predict_api
#include <mxnet/c_predict_api.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Read file to buffer
class BufferFile {
public:
  std::string file_path_;
  int length_;
  char* buffer_;

  explicit BufferFile(std::string file_path)
    :file_path_(file_path) {

    std::ifstream ifs(file_path.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
      std::cerr << "Can't open the file. Please check " << file_path << ". \n";
      length_ = 0;
      buffer_ = NULL;
      return;
    }

    ifs.seekg(0, std::ios::end);
    length_ = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::cout << file_path.c_str() << " ... " << length_ << " bytes\n";

    buffer_ = new char[sizeof(char) * length_];
    ifs.read(buffer_, length_);
    ifs.close();
  }

  int GetLength() {
    return length_;
  }
  char* GetBuffer() {
    return buffer_;
  }

  ~BufferFile() {
    if (buffer_) {
      delete[] buffer_;
      buffer_ = NULL;
    }
  }
};

// LoadSynsets
// Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
std::vector<std::string> LoadSynset(std::string synset_file) {
  std::ifstream fi(synset_file.c_str());

  if (!fi.is_open()) {
    std::cerr << "Error opening synset file " << synset_file << std::endl;
    assert(false);
  }

  std::vector<std::string> output;

  std::string synset, lemma;
  while (fi >> synset) {
    getline(fi, lemma);
    output.push_back(lemma);
  }

  fi.close();

  return output;
}

class MXInferencer
{
public:  
  enum MXTrainedImgFormat { eGrayScale, eRGB, eBGR };
  MXInferencer(int dev_type, int dev_id) 
    : infer_hnd_(NULL), output_data_(NULL) {
    output_size_ = 0;
    dev_type_ = dev_type;
    dev_id_ = dev_id;

    use_mean_file_ = false;
    nd_hnd_ = NULL;
    nd_data_ = NULL;

    use_mean_value_ = false;
    mean_value_[0] = 0;
    mean_value_[1] = 0;
    mean_value_[2] = 0;
  };
  ~MXInferencer() {};

public:
  int Initialize(std::string &json_file, std::string &param_file,
    std::vector<std::string> &input_nodes_name, std::vector<int> &input_shape, 
    MXTrainedImgFormat trained_img_format, float input_scale = 1.0);
  int CleanUp();

  int SetInputMean(float *mean_value = NULL, const char *nd_file = NULL);

  int Inference(cv::Mat src);
  
  int GetNetOutput(int output_index, float **output_data, int *output_size);
  int GetOutClassInfo(std::string &synset_file, float *data, int data_size);

private:
  enum MXShapeIndex { eNum, eChannel, eHeight, eWidth };

private:
  int dev_id_;
  int dev_type_;

  float input_scale_;

  PredictorHandle infer_hnd_;

  NDListHandle nd_hnd_;
  // The memory in it is mallocated by MXNDListCreate, and released by MXNDListFree.
  const mx_float* nd_data_;
  bool use_mean_file_;

  float mean_value_[3];
  bool use_mean_value_;

  // Shape: (num, channel, height, width)
  std::vector<int> inshape_;  
  int input_size_;

  // 
  mx_float *output_data_;
  int output_size_;

  //
  cv::Mat input_img_;
  cv::Mat src_ch_restrict_;
  MXTrainedImgFormat trained_img_format_;
};

int MXInferencer::Initialize(std::string &json_file, std::string &param_file, 
  std::vector<std::string> &input_nodes_name, std::vector<int> &input_shape, 
  MXTrainedImgFormat trained_img_format, float input_scale) {

  if (input_shape.size() != 4) {
    return -1;
  }

  BufferFile json_data(json_file);
  BufferFile param_data(param_file);

  if (json_data.GetLength() == 0 ||
    param_data.GetLength() == 0) {
    return -1;
  }

  mx_uint num_input_nodes = input_nodes_name.size();

  //const char* input_key[1] = { "data" };
  //const char** input_keys = input_key;
  const char **input_keys = new const char *[num_input_nodes];
  for (int i = 0; i < num_input_nodes; i++) {
    input_keys[i] = input_nodes_name[i].c_str();
  }

  trained_img_format_ = trained_img_format;
  input_scale_ = input_scale;
  inshape_.assign(input_shape.begin(), input_shape.end()); 
  if ((trained_img_format_ == eGrayScale && inshape_[eChannel] != 1) ||
    (trained_img_format_ != eGrayScale && inshape_[eChannel] != 3) ||
    input_scale_ == 0) {
    std::cerr << "Error: Please check the input params!" << std::endl;
    return -1;
  }

  input_size_ = inshape_[0] * inshape_[1] * inshape_[2] * inshape_[3];

  const mx_uint input_shape_indptr[2] = { 0, 4 };
  const mx_uint input_shape_data[4] = { 
    static_cast<mx_uint>(inshape_[eNum]),
    static_cast<mx_uint>(inshape_[eChannel]),
    static_cast<mx_uint>(inshape_[eHeight]),
    static_cast<mx_uint>(inshape_[eWidth]) };

  // Create Predictor
  MXPredCreate((const char*)json_data.GetBuffer(),
    (const char*)param_data.GetBuffer(),
    static_cast<size_t>(param_data.GetLength()),
    dev_type_,
    dev_id_,
    num_input_nodes,
    input_keys,
    input_shape_indptr,
    input_shape_data,
    &infer_hnd_);

  delete[]input_keys;

  if (infer_hnd_ == NULL) {
    std::cerr << "Error: MXPredCreate Failed!" << std::endl;
    return -1;
  }

  return 0;
}

int MXInferencer::CleanUp() {
  // Release NDList
  if (nd_hnd_)
    MXNDListFree(nd_hnd_);

  if (output_data_)
    free(output_data_);

  // Release Predictor
  if (infer_hnd_)
    MXPredFree(infer_hnd_);

  return 0;
}

int MXInferencer::SetInputMean(float *mean_value, const char *nd_file) {
  if ((mean_value == NULL && nd_file == NULL) ||
    (mean_value != NULL && nd_file != NULL)) {
    printf("Error: Please check the input params! \n");
    return -1;
  }

  if (infer_hnd_ == NULL) {
    printf("Error: The PredictorHandle handle is invalid! \n");
    return -1;
  }

  if (mean_value != NULL) {
    for (int i = 0; i < inshape_[eChannel]; i++) {
      mean_value_[i] = mean_value[i];
    }
    use_mean_value_ = true;
  }
  else {
    if (nd_hnd_ != NULL) {
      MXNDListFree(nd_hnd_);
      nd_hnd_ = NULL;
    }
    BufferFile nd_buf(nd_file);

    if (nd_buf.GetLength() <= 0) {
      printf("Error: Mean file is empty!\n");
      return -1;
    }
    mx_uint nd_index = 0;
    mx_uint nd_len;
    const mx_uint* nd_shape = 0;
    const char* nd_key = 0;
    mx_uint nd_ndim = 0;

    MXNDListCreate((const char*)nd_buf.GetBuffer(),
      nd_buf.GetLength(),
      &nd_hnd_, &nd_len);

    if (nd_hnd_ == NULL) {
      return -1;
    }

    MXNDListGet(nd_hnd_, nd_index, &nd_key, &nd_data_, &nd_shape, &nd_ndim);

    use_mean_file_ = true;
  }

  return 0;
}

// 扩展可选无均值输入
int MXInferencer::Inference(cv::Mat src) {  
  if (src.empty()) {
    std::cerr << "Error: Input image is empty!\n";
    return -1;
  }
  // Read Image Data
  std::vector<mx_float> image_data = std::vector<mx_float>(input_size_);

  // Convert channel and size.
  if (src.channels() == 3 && inshape_[eChannel] == 1)
    cv::cvtColor(src, src_ch_restrict_, cv::COLOR_BGR2GRAY);
  else if (src.channels() == 4 && inshape_[eChannel] == 1)
    cv::cvtColor(src, src_ch_restrict_, cv::COLOR_BGRA2GRAY);
  else if (src.channels() == 4 && inshape_[eChannel] == 3)
    cv::cvtColor(src, src_ch_restrict_, cv::COLOR_BGRA2BGR);
  else if (src.channels() == 1 && inshape_[eChannel] == 3)
    cv::cvtColor(src, src_ch_restrict_, cv::COLOR_GRAY2BGR);
  else
    src_ch_restrict_ = src;

  resize(src_ch_restrict_, input_img_, cv::Size(inshape_[eWidth], inshape_[eHeight]));

  // Preprocess.
  if (use_mean_file_) { 
    mx_float* ptr_image_c0 = image_data.data();
    mx_float* ptr_image_c1 = image_data.data() + input_size_ / 3;
    mx_float* ptr_image_c2 = image_data.data() + input_size_ / 3 * 2;

    const mx_float* ptr_mean_c0 = nd_data_;
    const mx_float* ptr_mean_c1 = nd_data_ + input_size_ / 3;
    const mx_float* ptr_mean_c2 = nd_data_ + input_size_ / 3 * 2;

    for (int i = 0; i < input_img_.rows; i++) {
      uchar* data = input_img_.ptr<uchar>(i);
      if (inshape_[eChannel] == 1) {
        for (int j = 0; j < input_img_.cols; j++) {
          *ptr_image_c0++ = data[j] - *ptr_mean_c0++;
        }
      }
      else if (inshape_[eChannel] == 3) {
        for (int j = 0; j < input_img_.cols; j++) {
          *ptr_image_c1++ = data[j * 3 + 1] - *ptr_mean_c1++;
          // The BGR format is read in opencv.
          if (trained_img_format_ == eBGR) {
            *ptr_image_c0++ = data[j * 3 + 0] - *ptr_mean_c0++;
            *ptr_image_c2++ = data[j * 3 + 2] - *ptr_mean_c2++;
          }
          else if (trained_img_format_ == eRGB) {
            *ptr_image_c0++ = data[j * 3 + 2] - *ptr_mean_c0++;
            *ptr_image_c2++ = data[j * 3 + 0] - *ptr_mean_c2++;
          }
        }
      }
    }  
  }
  else if (use_mean_value_) {
    if (inshape_[eChannel] == 1) {
      for (int i = 0; i < image_data.size(); i++) {
        image_data[i] -= mean_value_[0];
      }
    }
    else if (inshape_[eChannel] == 3) {
      mx_float* ptr_image_c0 = image_data.data();
      mx_float* ptr_image_c1 = image_data.data() + input_size_ / 3;
      mx_float* ptr_image_c2 = image_data.data() + input_size_ / 3 * 2;

      for (int i = 0; i < input_img_.rows; i++) {
        for (int j = 0; j < input_img_.cols; j++) {
          *ptr_image_c0++ -= mean_value_[0];
          *ptr_image_c1++ -= mean_value_[1];
          *ptr_image_c2++ -= mean_value_[2];
        }
      }
    }
  }
  // *input_scale

  // Set Input Image
  MXPredSetInput(infer_hnd_, "data", image_data.data(), input_size_);

  // Do Predict Forward
  MXPredForward(infer_hnd_);

  return 0;
}

// 还要考虑多个输出的情况
int MXInferencer::GetNetOutput(int output_index, float **output_data, int *output_size) {

  if (output_data_ == NULL) {
    mx_uint *shape = 0;
    mx_uint shape_len;

    // Get Output Result
    MXPredGetOutputShape(infer_hnd_, output_index, &shape, &shape_len);

    output_size_ = 1;
    for (mx_uint i = 0; i < shape_len; ++i)
      output_size_ *= shape[i];

    output_data_ = (mx_float*)malloc(sizeof(mx_float) * output_size_);
  }

  *output_size = output_size_;
  MXPredGetOutput(infer_hnd_, output_index, output_data_, output_size_);

  *output_data = output_data_;
  
  return 0;
}

int MXInferencer::GetOutClassInfo(std::string &synset_file, float *data, int data_size) {
  // Synset path for your model, you have to modify it
  std::vector<std::string> synset = LoadSynset(synset_file);

  if (data_size != synset.size()) {
    std::cerr << "Error: Result data and synset size does not match!" << std::endl;
    return -1;
  }

  float best_accuracy = 0.0;
  int best_idx = 0;

  for (int i = 0; i < data_size; i++) {
    printf("Accuracy[%d] = %.8f\n", i, data[i]);

    if (data[i] > best_accuracy) {
      best_accuracy = data[i];
      best_idx = i;
    }
  }

  printf("Best Result: [%s] id = %d, accuracy = %.8f\n",
    synset[best_idx].c_str(), best_idx, best_accuracy);
  
  return 0;
}

int main(int argc, char* argv[]) {

  std::string test_file = "D://Documents//GitHub//test-material//pic//2.jpg";

  // Models path for your model, you have to modify it
  //std::string json_file = "D://Documents//GitHub//test-material//model//Inception//resnet-18-0000.json";
  //std::string param_file = "D://Documents//GitHub//test-material//model//Inception//resnet-18-0000.params";
  std::string json_file = "D://Documents//GitHub//test-material//model//vgg//vgg16-0000.json";
  std::string param_file = "D://Documents//GitHub//test-material//model//vgg//vgg16-0000.params";

  std::string synset_file = "D://Documents//GitHub//test-material//model//Inception//synset.txt";
  std::string nd_file = "D://Documents//GitHub//test-material//model//Inception//mean_224.nd";
  
  int dev_type = 2; // 1: cpu, 2: gpu
  int dev_id = 0;
  MXInferencer *mx_infer = new MXInferencer(dev_type, dev_id);
  
  std::vector<std::string> input_nodes_name;
  input_nodes_name.push_back("data");
  std::vector<int> input_shape;
  input_shape.push_back(1);   // num
  input_shape.push_back(3);   // channel
  input_shape.push_back(224); // height
  input_shape.push_back(224); // width
  float input_scale = 1.0;

  mx_infer->Initialize(json_file, param_file, input_nodes_name, input_shape, mx_infer->eRGB, input_scale);

  mx_infer->SetInputMean(NULL, nd_file.c_str());
  
  cv::Mat src = cv::imread(test_file);
  mx_infer->Inference(src);

  int output_index = 0;
  float *output_data = NULL;
  int output_size = 0;
  mx_infer->GetNetOutput(output_index, &output_data, &output_size);

  mx_infer->GetOutClassInfo(synset_file, output_data, output_size);

  mx_infer->CleanUp();

  delete mx_infer;

  return 0;
}
#endif