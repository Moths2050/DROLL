#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

//
int number_of_projections;
int col_count;
int row_count;
float col_spacing; // The spacing of the image column
float row_spacing; // The spacing of the image row
float col_orign;  // The orign of the image column
float row_orign;  // The orign of the image row
int detector_shape; // The length of the detector
float detector_spacing; // The spacing of the detector
float detector_origin;  // The orign of the detector
float sid; // source_isocenter_distance
float sdd; // source_detector_distance

// Aw_cu
void FBP_2D_Aw_cu(at::Tensor input, const float * volume_ptr, float * out, //const float * ray_vectors,
                  const int number_of_projections, const int volume_size_x, const int volume_size_y, const float volume_spacing_x, const float volume_spacing_y,
                  const float volume_origin_x, const float volume_origin_y, const int detector_size, const float detector_spacing, const float detector_origin,
                  const float sid, const float sdd);
// Atw_cu
void FBP_2D_Atw_cu(at::Tensor input, const float * sinogram_ptr, float * out, //const float * ray_vectors,
                  const int number_of_projections, const int volume_size_x, const int volume_size_y, const float volume_spacing_x, const float volume_spacing_y,
                  const float volume_origin_x, const float volume_origin_y, const int detector_size, const float detector_spacing, const float detector_origin,
                  const float sid, const float sdd);

void ShowInfo(int number_of_projections, int col_count, int row_count, float col_spacing, float row_spacing,
              float col_orign, float row_orign, int detector_shape, float detector_spacing, float detector_origin,
              float sid, float sdd){
    printf("The number of the projections is:    %d\n", number_of_projections);
    printf("The number of the image column is:   %d\n", col_count);
    printf("The number of the image row is:      %d\n", row_count);
    printf("The sapcing of the image column is:  %f\n", col_spacing);
    printf("The sapcing of the image row is:     %f\n", row_spacing);
    printf("The orign of the image column is:    %f\n", col_orign);
    printf("The orign of the image row is:       %f\n", row_orign);
    printf("The number of the detector is:       %d\n", detector_shape);
    printf("The sapcing of the detector is:      %f\n", detector_spacing);
    printf("The orign of the detector is:        %f\n", detector_origin);
    printf("Thesource_isocenter_distance is:     %f\n", sid);
    printf("Thesource_detector_distance is:      %f\n", sdd);
}

void FBP_Allocate_cu(const float * ray_vectors, const int volume_size_x,  const int volume_size_y, const int detector_size, const int number_of_projections);
void FBP_Free_cu();

void FBP_Allocate(at::Tensor central_ray_vectors, at::Tensor input_int, at::Tensor input_float){
    float * ray_vectors = central_ray_vectors[0].data_ptr<float>();

    int   * pti = input_int[0].data_ptr<int>();
    float * ptf = input_float[0].data_ptr<float>();

    // 4)
    number_of_projections = pti[0];
    // 5)
    col_count = pti[1];
    // 6)
    row_count = pti[2];

    // 7)
    col_spacing = ptf[0]; // The spacing of the image column
    row_spacing = ptf[1]; // The spacing of the image row

    // 8)
    col_orign = ptf[2];  // The orign of the image column
    row_orign = ptf[3];  // The orign of the image row

    // 9)
    detector_shape = pti[3]; // The length of the detector
    // 10)
    detector_spacing = ptf[4]; // The spacing of the detector
    // 11)
    detector_origin = ptf[5];  // The orign of the detector
    // 12)
    sid = ptf[6]; // source_isocenter_distance
    //  13)
    sdd = ptf[7]; // source_detector_distance

    //ShowInfo(number_of_projections, col_count, row_count, col_spacing, row_spacing, col_orign, row_orign, detector_shape, detector_spacing, detector_origin, sid, sdd);

    FBP_Allocate_cu(ray_vectors, col_count,  row_count, detector_shape, number_of_projections);
}

void FBP_Free(){
    FBP_Free_cu();
}


void FBP_2D_Aw(at::Tensor input, at::Tensor output)
{
    //using namespace at;
    //int ibatch = input::sizes()[0];
    //int ichannel = input::size()[1];

    int ibatch = at::size(input, 0);
    int ichannel = at::size(input, 1);

    int i;
    int j;
    for(i = 0; i < ibatch; ++i)
    {
        for(j = 0; j < ichannel; ++j)
        {
            float * volume_ptr  = input[i][j].data_ptr<float>();
            // printf("volumn value: %f  %f  %f\n",volume_ptr[0], volume_ptr[1], volume_ptr[510]); // Can not printf the data on the --device--

            float * out         = output[i][j].data_ptr<float>();

            FBP_2D_Aw_cu(input, volume_ptr, out,
                     number_of_projections, col_count, row_count, col_spacing, row_spacing,
                     col_orign, row_orign, detector_shape, detector_spacing, detector_origin,
                     sid, sdd);
        }
    }
}

void FBP_2D_Atw(at::Tensor input, at::Tensor output)
{
    //using namespace at;
    //int ibatch = input::sizes()[0];
    //int ichannel = input::size()[1];

    int ibatch = at::size(input, 0);
    int ichannel = at::size(input, 1);

    int i;
    int j;
    for(i = 0; i < ibatch; ++i)
    {
        for (j = 0; j < ichannel; ++j)
        {
            float * sinogram_ptr  = input[i][j].data_ptr<float>();
            // printf("volumn value: %f  %f  %f\n",volume_ptr[0], volume_ptr[1], volume_ptr[510]); // Can not printf the data on the --device--

            float * out         = output[i][j].data_ptr<float>();

            FBP_2D_Atw_cu(input, sinogram_ptr, out,
                      number_of_projections, col_count, row_count, col_spacing, row_spacing,
                      col_orign, row_orign, detector_shape, detector_spacing, detector_origin,
                      sid, sdd);
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FBP_cuda_lib";
    m.def("FBP_Allocate",    &FBP_Allocate,    "FBP_Allocate (CUDA)");
    m.def("FBP_Free",        &FBP_Free,        "FBP_Free (CUDA)");
    m.def("FBP_2D_Aw",       &FBP_2D_Aw,       "FBP_2D_Aw (CUDA)");
    m.def("FBP_2D_Atw",      &FBP_2D_Atw,      "FBP_2D_Atw (CUDA)");

}
