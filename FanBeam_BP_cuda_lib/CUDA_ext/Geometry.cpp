#include <torch/torch.h>
#include <torch/extension.h>
#include <vector>

#include <pybind11/pybind11.h>
namespace py = pybind11;


#include <string>

class student{
public:
    student(){};
    ~student(){};

public:
    std::string name;
    int Chinese;
    int Mathematics;
    int English;
    float total;
};


void PS(class student & st)
{
    printf("%s\n", st.name);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FBP_cuda_lib";
    py::class_<student>(m, "student")
        .def_readwrite("name", &student::name)
        .def_readwrite("Chinese", &student::Chinese)
        .def_readwrite("Mathematics", &student::Mathematics)
        .def_readwrite("English", &student::English)
        .def_readwrite("total", &student::total);

    //m.def("FBP_2D_Aw", &FBP_2D_Aw, "FBP_2D_Aw (CUDA)");
    m.def("PS", &PS);
}