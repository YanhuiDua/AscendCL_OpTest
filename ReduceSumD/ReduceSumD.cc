#include <iostream>
#include <vector>

#include "common/nputensor.h"

int main() {
  // Init
  ACL_CALL(aclInit(nullptr));
  ACL_CALL(aclrtSetDevice(0));

  // Get Run Mode - ACL_HOST
  aclrtRunMode runMode;
  ACL_CALL(aclrtGetRunMode(&runMode));
  std::string run_mode_str = (runMode == ACL_DEVICE) ? "ACL_DEVICE" : "ACL_HOST";
  std::cout << "aclrtRunMode is : " << run_mode_str << std::endl;

  // op type
  const std::string op_type = "ReduceSumD";

  // input - X
  const std::vector<int64_t> x1_dims{1, 3, 76, 156};
  std::vector<float> x1_data(35568);
  std::iota(x1_data.begin(), x1_data.end(), 0);
  // attr
  const bool keep_dims = false;
  const std::vector<int64_t> axes = {2,3};
  // output - y
  const std::vector<int64_t> y_dims{1,3};

  // input - x
  auto x1 = new npuTensor<float>(ACL_FLOAT, x1_dims.size(), x1_dims.data(), ACL_FORMAT_NCHW, x1_data.data());

  // set inputs desc and buffer
  std::vector<aclTensorDesc *> input_descs;
  std::vector<aclDataBuffer *> input_buffers;
  input_descs.emplace_back(x1->desc);
  input_buffers.emplace_back(x1->buffer);

  // output - y
  auto y = new npuTensor<float>(ACL_FLOAT, y_dims.size(), y_dims.data(), ACL_FORMAT_NCHW, nullptr);

  // set output desc and buffer
  std::vector<aclTensorDesc *> output_descs;
  std::vector<aclDataBuffer *> output_buffers;
  output_descs.emplace_back(y->desc);
  output_buffers.emplace_back(y->buffer);

  // attr
  auto attr = aclopCreateAttr();
  ACL_CALL(aclopSetAttrListInt(attr, "axes", axes.size(), axes.data()));
  ACL_CALL(aclopSetAttrBool(attr, "keep_dims", keep_dims));
  
  // create stream
  aclrtStream stream = nullptr;
  ACL_CALL(aclrtCreateStream(&stream));

  std::cout << "aclopCompileAndExecute : " << op_type << std::endl;
  ACL_CALL(aclopCompileAndExecute(op_type.c_str(), 
            input_descs.size(), input_descs.data(), input_buffers.data(), 
            output_descs.size(), output_descs.data(), output_buffers.data(), 
            attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, stream));

  // sync and destroy stream
  ACL_CALL(aclrtSynchronizeStream(stream));
  ACL_CALL(aclrtDestroyStream(stream));

  // print output
  // x1->Print("x1");
  y->Print("y");

  // destroy - inputs
  x1->Destroy();
  // destroy - outputs
  y->Destroy();

  aclopDestroyAttr(attr);

  // release
  ACL_CALL(aclrtResetDevice(0));
  ACL_CALL(aclFinalize());

  return 0;
}