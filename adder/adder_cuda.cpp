#include <torch/torch.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <torch/csrc/THP_export.h>
#include <torch/csrc/utils/pybind.h>

int adder_cuda_forward(         //定义forward参数(cuda)
    const at::Tensor &input,
    const at::Tensor &weight, 
    // const at::Tensor &bias,
    at::Tensor &output,
    int KW, int KH,
    int SW, int SH,
    int PW, int PH);

int adder_cuda_backward_grad_in( //定义对X的backward参数(cuda)
    at::Tensor &grad_out,
    at::Tensor &input,
    at::Tensor &weight,
    at::Tensor &grad_in,
    int KW, int KH,
    int SW, int SH,
    int PW, int PH);

int adder_cuda_backward_grad_weight( //定义对W的backward参数(cuda)
    at::Tensor &grad_out,
    at::Tensor &input,
    at::Tensor &weight,
    at::Tensor &grad_weight,
    int KW, int KH,
    int SW, int SH,
    int PW, int PH);

#define CHECK_CUDA(x) AT_ASSERT((x).type().is_cuda(), #x "must be a CUDA tensor")        //定义不用管
#define CHECK_CONTIGUOUS(x) AT_ASSERT((x).type().is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA((x));   \
    CHECK_CONTIGUOUS((x))
 
int adder_forward(               //定义forward参数 
    const at::Tensor &input,
    const at::Tensor &weight,
    // const at::Tensor &bias,
    at::Tensor &output,
    int KW, int KH,       //kernel_size
    int SW, int SH,       //stride
    int PW, int PH)       //padding
{
    // TODO: add checks checks
    return adder_cuda_forward(   //返回forward参数
        input,
        weight,
        // bias,
        output,
        KW, KH,
        SW, SH,
        PW, PH);
}

int adder_backward_input(     //定义对X的background参数
    at::Tensor &grad_out,
    at::Tensor &input,
    at::Tensor &weight,
    at::Tensor &grad_in,
    int KW, int KH,
    int SW, int SH,
    int PW, int PH)
{
    // TODO: add checks checks
    return adder_cuda_backward_grad_in(  //返回X的background参数
        grad_out,  
        input,
        weight,
        grad_in,
        KW, KH,
        SW, SH,
        PW, PH);
}

int adder_backward_weight(  //定义对W的background参数
    at::Tensor &grad_out,
    at::Tensor &input,
    at::Tensor &weight,
    at::Tensor &grad_weight,
    int KW, int KH,
    int SW, int SH,
    int PW, int PH)
{
    // TODO: add checks checks
    return adder_cuda_backward_grad_weight( //定义对W的background参数
        grad_out,
        input,
        weight,
        grad_weight,
        KW, KH,
        SW, SH,
        PW, PH);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)   //固定格式
{
    m.def("forward", &adder_forward, "adder forward (CUDA)");
    m.def("backward_input", &adder_backward_input, "adder backward input (CUDA)");
    m.def("backward_weight", &adder_backward_weight, "adder backward weight (CUDA)");
}
